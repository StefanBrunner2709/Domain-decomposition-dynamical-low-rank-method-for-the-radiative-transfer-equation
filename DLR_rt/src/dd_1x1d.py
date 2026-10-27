"""  
Domain decomposition for the 1x1d radiative transfer equation, serial version


Contains everything that the serial and MPI version share: setup of the subdomains,
the local time step on one subdomain, timings and the result of a run
The MPI version (dd_mpi.py) only replaces where the neighbour rows come from
"""
from contextlib import contextmanager # for timing and memory usage
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

    Note that memory is only measured if tracemalloc is running
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
            mem_phase_end(phase, self.mem_peak, m0)
            self.times[phase] = self.times.get(phase, 0.0) + perf_counter() - t0

@dataclass
class RunResult:
    """ 
    Result of a run on one or more subdomains, ordered left to right

    Serial: all subdomains. MPI: only the subdomain of this rank
    use then dd_mpi.gather_result to get all of them on rank 0
    """
    lrs: list[LR] # low-rank objects
    grids: list[Grid_1x1d] # subdomain grids
    time: list[float] # time of each snapshot
    rank_hist: list[list[int]] # rank of each subdomain at each snapshot
    mem_peak: list[dict] # [domain] -> {phase: peak memory in bytes}
    snap_t: list[float] = field(default_factory=list) # time of each snapshot
    snap_f: list[np.ndarray] = field(default_factory=list) # f at each snapshot
    runtime: float = 0.0 # total runtime of the run

    @property
    def n_domains(self) -> int:
        return len(self.grids)

    def f(self) -> np.ndarray:
        """ 
        Full matrix f on all subdomains of this result glued together in x
        """
        return np.concatenate([full_f(lr) for lr in self.lrs], axis=0)

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
    Full matrix f = U S V^T of a low-rank object
    """
    return lr.U @ lr.S @ lr.V.T

def setup_domains(Nx: int, Nmu: int, r: int, n_domains: int, seed: int = 0, coeffs=None):
    """  
    Split grid, initial conditions and one random generator per subdomain

    Every subdomain is built, also the MPI version on every rank so that the initial
    conditions does not depend on the number of ranks
    
    Returns
    -------
    grids, lr0s, rngs
        List of length n_domains, ordered left to right
    """
    # Initial condition uses global np.random
    np.random.seed(seed)
    grids = Grid_1x1d(Nx, Nmu, r).split_n(n_domains, _coeffs=coeffs)
    lr0s = [setInitialCondition_1x1d_lr(g) for g in grids]
    rngs = [np.random.default_rng([seed, i]) for i in range(n_domains)]
    return grids, lr0s, rngs

def local_step(lr: LR, grid: Grid_1x1d, F_b: np.ndarray, t: float, dt: float,
               rng, tol_sing_val: float, drop_tol: float, timer: PhaseTimer,
               option: str = "lie"):
    """ 
    One time step on one subdomain: add basis -> PSI -> drop basis
    """