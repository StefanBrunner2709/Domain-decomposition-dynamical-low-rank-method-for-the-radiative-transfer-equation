"""
Domain decomposition for the 1x1d radiative transfer equation, serial version.

Contains everything that the serial and MPI version share: setup of the subdomains,
the local time step on one subdomain, timings and the result of a run.
The MPI version (dd_mpi.py) only replaces where the neighbour rows come from.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm import tqdm

from DLR_rt.src.grid import Grid_1x1d
from DLR_rt.src.initial_condition import setInitialCondition_1x1d_lr
from DLR_rt.src.integrators import PSI_lie, PSI_strang
from DLR_rt.src.lr import (
    LR,
    add_basis_functions,
    computeF_b_edges,
    drop_basis_functions,
    first_row,
    last_row,
)
from DLR_rt.src.util import mem_phase_end, mem_phase_start


class PhaseTimer:
    """
    Accumulate time (and memory) per phase of the time step.

    Usage: with timer("PSI"): ...
    Note that memory is only measured if tracemalloc is running.
    """

    def __init__(self):
        self.times = {}
        self.mem_peak = {}

    @contextmanager
    def __call__(self, phase: str):
        """
        Context manager to measure time and memory of a phase.
        """
        self.mem_peak.setdefault(phase, 0)
        t0 = perf_counter()
        m0 = mem_phase_start()
        try:
            yield
        finally:
            mem_phase_end(self.mem_peak, phase, m0)
            self.times[phase] = self.times.get(phase, 0.0) + perf_counter() - t0


@dataclass
class RunResult:
    """
    Result of a run on one or more subdomains, ordered left to right.

    Serial: all subdomains. MPI: only the subdomain of this rank,
    use dd_mpi.gather_result to get all of them on rank 0.
    """
    lrs: list[LR]  # low-rank objects
    grids: list[Grid_1x1d]  # subdomain grids
    time: list[float]  # time after each time step, starts with 0
    rank_hist: list[list[int]]  # [domain][time step] rank of each subdomain
    timings: list[dict]  # [domain] -> {phase: seconds}
    mem_peak: list[dict]  # [domain] -> {phase: peak memory in bytes}
    snap_t: list[float] = field(default_factory=list)  # time of each snapshot
    snap_f: list[np.ndarray] = field(default_factory=list)  # f at each snapshot
    runtime: float = 0.0  # wall time of the time loop in seconds
    owner: list[int] = field(default_factory=list)  # MPI: [domain] -> rank
    rank_timings: list[dict] = field(default_factory=list)  # MPI: [rank] -> exchange

    @property
    def n_domains(self) -> int:
        return len(self.grids)

    def f(self) -> np.ndarray:
        """
        Full matrix f on all subdomains of this result glued together in x.
        """
        return np.concatenate([full_f(lr) for lr in self.lrs], axis=0)

    def timings_per_rank(self) -> list[dict]:
        """Timings summed over the domains of each MPI rank, plus its exchange.

        Serial: one entry, the sum over all domains.
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


def full_f(lr: LR) -> np.ndarray:
    """
    Full matrix f = U S V^T of a low-rank object.
    """
    return lr.U @ lr.S @ lr.V.T


def setup_domains(Nx: int, Nmu: int, r: int, n_domains: int, seed: int = 0,
                  coeffs=None):
    """
    Split grid, initial conditions and one random generator per subdomain.

    Every subdomain is built, in the MPI version on every rank, so that the initial
    condition does not depend on the number of ranks.

    Parameters
    ----------
    Nx : int
        Number of x grid points in the full domain.
    Nmu : int
        Number of mu grid points in the full domain.
    r : int
        Initial rank of the low-rank objects.
    n_domains : int
        Number of subdomains to split the full domain into.
    seed : int
        Seed for the global random generator, used for the initial condition.
    
    Returns
    -------
    grids, lr0s, rngs
        Lists of length n_domains, ordered left to right.
    """
    # Initial condition uses global np.random
    np.random.seed(seed)
    grids = Grid_1x1d(Nx, Nmu, r).split_n(n_domains, _coeffs=coeffs)
    lr0s = [setInitialCondition_1x1d_lr(g) for g in grids]
    # Own generator per subdomain for add_basis_functions, independent of
    # the order in which the domains are updated (serial vs MPI)
    rngs = [np.random.default_rng([seed, i]) for i in range(n_domains)]
    return grids, lr0s, rngs


def local_step(lr: LR, grid: Grid_1x1d, F_b: np.ndarray, t: float, dt: float,
               rng, tol_sing_val: float, drop_tol: float, timer: PhaseTimer,
               option: str = "lie"):
    """
    One time step on one subdomain: add basis -> PSI -> drop basis.

    Same for serial and MPI, only F_b has to be computed before.
    """
    with timer("add_basis"):
        lr, grid = add_basis_functions(lr, grid, F_b, tol_sing_val, rng=rng)

    with timer("PSI"):
        if option == "lie":
            lr, grid, _, _ = PSI_lie(lr, grid, dt, F_b, option_scheme="upwind")
        elif option == "strang":
            lr, grid = PSI_strang(lr, grid, dt, t, F_b)
        else:
            raise ValueError(f"Unknown option {option}, use 'lie' or 'strang'")

    with timer("drop_basis"):
        lr, grid = drop_basis_functions(lr, grid, drop_tol)

    return lr, grid

def block_step(lrs: list[LR], grids: list[Grid_1x1d], rngs: list,
               timers: list[PhaseTimer], t: float, dt: float,
               tol_sing_val: float, drop_tol: float, option: str = "lie",
               use_neighbours: bool = True,
               outer_left: np.ndarray | None = None,
               outer_right: np.ndarray | None = None):
    """
    One time step on a block of neighbouring subdomains, ordered left to right.

    Serial: the block is the whole domain, outer_left/outer_right are None.
    MPI: the block is the part of one rank, outer_left/outer_right are the rows
    received from the neighbouring ranks (None at the physical boundary).
    lrs and grids are updated in place.
    """
    n = len(lrs)

    # 1) Get boundary values of all domains, from the state before the step
    # Inside the block the rows come from neighbouring lr
    F_bs = []
    for i in range(n):
        with timers[i]("F_b"):
            if use_neighbours:
                inflow_left = last_row(lrs[i - 1]) if i > 0 else outer_left
                inflow_right = first_row(lrs[i + 1]) if i < n - 1 else outer_right
            else:
                inflow_left = inflow_right = None
            F_bs.append(computeF_b_edges(t, lrs[i], grids[i],
                                         inflow_left, inflow_right))

    # 2) local update, independent between the domains
    for i in range(n):
        lrs[i], grids[i] = local_step(
            lrs[i], grids[i], F_bs[i], t, dt, rngs[i],
            tol_sing_val, drop_tol, timers[i], option
        )


def integrate_serial(lrs: list[LR], grids: list[Grid_1x1d], rngs: list,
                     t_f: float, dt: float, tol_sing_val: float = 1e-6,
                     drop_tol: float = 1e-6, option: str = "lie",
                     use_neighbours: bool = True, snap_every: int | None = None,
                     progress: bool = True) -> RunResult:
    """
    Serial domain decomposition over n subdomains, ordered from left to right.

    With one subdomain this is the simulation without decomposition (dlr_inflow).
    The lr objects and grids are changed in place.

    Parameters
    ----------
    use_neighbours : bool
        If True, the subdomains get the inflow from their neighbours' edge rows.
        If False (negative control), every subdomain uses tanh(t) on both sides.
    snap_every : int or None
        If given, f on all subdomains is stored every snap_every steps and at t_f.
    """
    n = len(lrs)
    lrs, grids = list(lrs), list(grids)
    timers = [PhaseTimer() for _ in range(n)]

    t = 0.0
    step = 0
    time = [t]
    rank_hist = [[g.r] for g in grids]
    snap_t, snap_f = [], []

    def snapshot():
        snap_t.append(t)
        snap_f.append(np.concatenate([full_f(lr) for lr in lrs], axis=0))

    if snap_every:
        snapshot()

    t_start = perf_counter()
    with tqdm(total=t_f / dt, desc="Running Simulation", disable=not progress) as pbar:
        while t < t_f:
            pbar.update(1)

            if t + dt > t_f:
                dt = t_f - t

            block_step(lrs, grids, rngs, timers, t, dt, tol_sing_val, drop_tol,
                       option, use_neighbours)

            t += dt
            step += 1
            time.append(t)
            for i in range(n):
                rank_hist[i].append(grids[i].r)

            if snap_every and (step % snap_every == 0 or t >= t_f):
                snapshot()

    return RunResult(
        lrs=lrs,
        grids=grids,
        time=time,
        rank_hist=rank_hist,
        timings=[tm.times for tm in timers],
        mem_peak=[tm.mem_peak for tm in timers],
        snap_t=snap_t,
        snap_f=snap_f,
        runtime=perf_counter() - t_start,
    )


if __name__ == "__main__":
    # Short demo, the checks are in DLR_rt/tests/test_dd_1x1d.py
    grids, lr0s, rngs = setup_domains(Nx=256, Nmu=64, r=5, n_domains=4, seed=0)
    res = integrate_serial(lr0s, grids, rngs, t_f=0.2, dt=1e-3,
                           tol_sing_val=1e-5, drop_tol=1e-5)
    print(f"{res.n_domains} domains, runtime {res.runtime:.2f} s, "
          f"final ranks {[h[-1] for h in res.rank_hist]}")
