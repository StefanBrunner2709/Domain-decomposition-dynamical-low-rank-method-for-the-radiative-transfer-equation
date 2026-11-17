"""
Subdomains of the 1x1d problem and the serial domain decomposition.

Everything the serial and the MPI version share: setup of the subdomains, one time
step on a block of neighbouring subdomains, the timers and the result of a run.
The MPI version (mpi.py) only adds where the edge rows of the block come from.

    grids, lrs, rngs = setup_domains(Nx=256, Nmu=64, r=5, n_domains=4)
    res = integrate_serial(lrs, grids, rngs, t_f=0.2, dt=1e-3, inflow=beam)

The rank of a subdomain is lr.r. grid.r is only the initial rank, the grids are
not changed during the run.
"""
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm import tqdm

from DLR_rt.src.d1x1.inflow import tanh
from DLR_rt.src.d1x1.lowrank import LR, add_basis, boundary_values, drop_basis, psi_step
from DLR_rt.src.grid import Grid_1x1d
from DLR_rt.src.initial_condition import setInitialCondition_1x1d_lr


class PhaseTimer:
    """
    Time (and peak memory) per phase of the time step, summed over all steps.

        with timer("PSI"): ...

    Memory is only measured if tracemalloc is running: the largest extra memory a
    phase needed on top of what was allocated when it started.
    """

    def __init__(self):
        self.times = {}
        self.mem_peak = {}

    @contextmanager
    def __call__(self, phase: str):
        self.mem_peak.setdefault(phase, 0)
        tracing = tracemalloc.is_tracing()
        if tracing:
            tracemalloc.reset_peak()
            mem0 = tracemalloc.get_traced_memory()[0]
        t0 = perf_counter()
        try:
            yield
        finally:
            if tracing:
                peak = tracemalloc.get_traced_memory()[1]
                self.mem_peak[phase] = max(self.mem_peak[phase], peak - mem0)
            self.times[phase] = self.times.get(phase, 0.0) + perf_counter() - t0


@dataclass
class RunResult:
    """
    Result of a run on one or more subdomains, ordered left to right.

    Serial: all subdomains. MPI: only the subdomains of this rank,
    mpi.gather_result collects all of them on rank 0.
    """
    lrs: list[LR]
    grids: list[Grid_1x1d]
    time: list[float]  # time after each step, starts with 0
    rank_hist: list[list[int]]  # [domain][step] low rank r, starts with r0
    timings: list[dict]  # [domain] -> {phase: seconds}
    mem_peak: list[dict]  # [domain] -> {phase: peak memory in bytes}
    snap_t: list[float] = field(default_factory=list)  # time of each snapshot
    snap_f: list[np.ndarray] = field(default_factory=list)  # full f at each snapshot
    runtime: float = 0.0  # wall time of the time loop in seconds
    owner: list[int] = field(default_factory=list)  # MPI: [domain] -> rank
    rank_timings: list[dict] = field(default_factory=list)  # MPI: [rank] -> exchange
    # MPI: [rank] -> time in block_step. With dynamic balancing the subdomains (and
    # their timers) move between ranks, then only this is the busy time of a rank
    rank_busy: list[float] = field(default_factory=list)
    balance: dict = field(default_factory=dict)  # MPI: Balancer.summary(), if used

    @property
    def n_domains(self) -> int:
        return len(self.grids)

    def f(self) -> np.ndarray:
        """Full f on all subdomains of this result glued together in x."""
        return np.concatenate([lr.full() for lr in self.lrs], axis=0)

    def timings_per_rank(self) -> list[dict]:
        """
        Timings summed over the subdomains of each MPI rank, plus its exchange.

        Serial: one entry, the sum over all subdomains.
        """
        owner = self.owner or [0] * self.n_domains
        per_rank = [dict(tim) for tim in self.rank_timings] or [{}]
        for o, tim in zip(owner, self.timings):
            for phase, seconds in tim.items():
                per_rank[o][phase] = per_rank[o].get(phase, 0.0) + seconds
        return per_rank

    def save(self, path: Path):
        """Save f, rank history and snapshots, load again with np.load."""
        np.savez(
            path,
            f=self.f(),
            X=np.concatenate([g.X for g in self.grids]),
            MU=self.grids[0].MU,
            time=np.array(self.time),
            rank_hist=np.array(self.rank_hist),
            snap_t=np.array(self.snap_t),
            snap_f=np.array(self.snap_f),
            runtime=self.runtime,
        )


def setup_domains(Nx: int, Nmu: int, r: int, n_domains: int, seed: int = 0,
                  coeffs: list[float] | None = None, sizes: list[int] | None = None):
    """
    Split the grid, initial condition and one random generator per subdomain.

    All subdomains are built, in the MPI version on every rank, so the initial
    condition does not depend on the number of ranks.

    coeffs: 1/epsilon per subdomain, None is the value of the full grid.
    sizes: grid points per subdomain left to right, sum Nx, None is an even split.

    Returns grids, lrs, rngs, lists of length n_domains ordered left to right.
    """
    np.random.seed(seed)  # the initial condition uses the global np.random
    grids = Grid_1x1d(Nx, Nmu, r).split_n(n_domains, _coeffs=coeffs, _sizes=sizes)
    lrs = []
    for g in grids:
        lr = setInitialCondition_1x1d_lr(g)
        lrs.append(LR(lr.U, lr.S, lr.V))
    # Own generator per subdomain for add_basis, so the random numbers do not depend
    # on the order in which the subdomains are updated (serial vs. MPI)
    rngs = [np.random.default_rng([seed, i]) for i in range(n_domains)]
    return grids, lrs, rngs


def block_step(lrs: list[LR], grids: list[Grid_1x1d], rngs: list,
               timers: list[PhaseTimer], t: float, dt: float,
               tol_sing_val: float, drop_tol: float, inflow=tanh,
               use_neighbours: bool = True,
               outer_left: np.ndarray | None = None,
               outer_right: np.ndarray | None = None):
    """
    One time step on a block of neighbouring subdomains, lrs is updated in place.

    Serial: the block is the whole domain. MPI: the part of one rank, outer_left and
    outer_right are the edge rows received from the neighbouring ranks, None at the
    physical boundary, where inflow(t, mu) -> (left, right) is used.

    use_neighbours=False is the negative control: every subdomain gets the physical
    inflow on both sides, as if it were alone.
    """
    n = len(lrs)

    # 1) Boundary values of all subdomains from the state before the step
    F_bs = []
    for i in range(n):
        with timers[i]("F_b"):
            left, right = inflow(t, grids[i].MU)
            if use_neighbours:
                if i > 0:
                    left = lrs[i - 1].last_row()
                elif outer_left is not None:
                    left = outer_left
                if i < n - 1:
                    right = lrs[i + 1].first_row()
                elif outer_right is not None:
                    right = outer_right
            F_bs.append(boundary_values(lrs[i], grids[i], left, right))

    # 2) Local update, independent between the subdomains
    for i in range(n):
        with timers[i]("add_basis"):
            lr = add_basis(lrs[i], grids[i], F_bs[i], tol_sing_val, rngs[i])
        with timers[i]("PSI"):
            lr = psi_step(lr, grids[i], F_bs[i], dt)
        with timers[i]("drop_basis"):
            lrs[i] = drop_basis(lr, grids[i], drop_tol)


def integrate_serial(lrs: list[LR], grids: list[Grid_1x1d], rngs: list,
                     t_f: float, dt: float, tol_sing_val: float = 1e-6,
                     drop_tol: float = 1e-6, inflow=tanh, use_neighbours: bool = True,
                     snap_every: int | None = None, progress: bool = True) -> RunResult:
    """
    Serial domain decomposition, subdomains ordered left to right.

    With one subdomain this is the simulation without decomposition.

    inflow: inflow(t, mu) -> (left, right) on the physical boundary, see inflow.py.
    use_neighbours: False is the negative control, see block_step.
    snap_every: store the full f every snap_every steps and at t_f.
    """
    n = len(lrs)
    lrs = list(lrs)
    timers = [PhaseTimer() for _ in range(n)]

    t = 0.0
    step = 0
    time = [t]
    rank_hist = [[lr.r] for lr in lrs]
    snap_t, snap_f = [], []

    def snapshot():
        snap_t.append(t)
        snap_f.append(np.concatenate([lr.full() for lr in lrs], axis=0))

    if snap_every:
        snapshot()

    t_start = perf_counter()
    with tqdm(total=t_f / dt, desc="Running Simulation", disable=not progress) as pbar:
        while t < t_f:
            pbar.update(1)
            if t + dt > t_f:
                dt = t_f - t

            block_step(lrs, grids, rngs, timers, t, dt, tol_sing_val, drop_tol,
                       inflow, use_neighbours)

            t += dt
            step += 1
            time.append(t)
            for i in range(n):
                rank_hist[i].append(lrs[i].r)
            if snap_every and (step % snap_every == 0 or t >= t_f):
                snapshot()

    return RunResult(
        lrs=lrs,
        grids=list(grids),
        time=time,
        rank_hist=rank_hist,
        timings=[tm.times for tm in timers],
        mem_peak=[tm.mem_peak for tm in timers],
        snap_t=snap_t,
        snap_f=snap_f,
        runtime=perf_counter() - t_start,
    )
