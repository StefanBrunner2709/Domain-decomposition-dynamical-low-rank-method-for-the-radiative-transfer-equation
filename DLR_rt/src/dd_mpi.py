""" 
Domain decomposition for the 1x1d radiative transfer equation, MPI version

One subdomain per rank, ordered from left to right by rank.
Per time step every rank only exchanges one row of length Nmu with each neighbour, the local update
is the same local step as in the serial version

Note: mpirun -n 4 python -m DLR_rt.src.dd_mpi
"""
import os 

# BLAS and LAPACK to 1 thread per MPI rank, has to be set before numpy is imported
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from time import perf_counter

import numpy as np
from mpi4py import MPI
from tqdm import tqdm

from DLR_rt.src.dd_1x1d import PhaseTimer, RunResult, full_f, local_step
from DLR_rt.src.grid import Grid_1x1d
from DLR_rt.src.lr import LR, computeF_b_edges, first_row, last_row


def neighbours(comm):
    """ 
    Ranks of the left and right neighbour, PROC_NULL at the physical boundary.

    Communication with PROC_NULL does nothing, so the first and last rank need no special 
    case
    """
    rank, size = comm.Get_rank(), comm.Get_size()
    left_nb = rank - 1 if rank > 0 else MPI.PROC_NULL # left neighbour, PROC_NULL at the left boundary
    right_nb = rank + 1 if rank < size - 1 else MPI.PROC_NULL # right neighbour, PROC_NULL at the right boundary
    return left_nb, right_nb

def exchange_edges(comm, lr: LR, left_nb, right_nb,
                   from_left: np.ndarray, from_right: np.ndarray):
    """ 
    Exchange the edge rows of f with both neighbours

    Sends the last row to the right and the first row to the left, receives the first row from the right and the last row from the left
    The received rows are stored in from_left and from_right, which must be preallocated with the correct shape
    """

    # Sendrecv sends and receives in one call
    comm.Sendrecv(last_row(lr), dest=right_nb, recvbuf=from_left, source=left_nb)
    comm.Sendrecv(first_row(lr), dest=left_nb, recvbuf=from_right, source=right_nb)

def integrate_mpi(lr: LR, grid: Grid_1x1d, rng, t_f: float, dt: float,
                  comm=MPI.COMM_WORLD, tol_sing_val: float = 1e-6,
                  drop_tol: float = 1e-6, option: str = "lie",
                  use_neighbours: bool = True, snap_every: int | None = None,
                  progress: bool = True) -> RunResult:
    """
    Domain decomposition with MPI, every rank integrates its own subdomain.

    All ranks have to call this together. Same parameters as
    dd_1x1d.integrate_serial, but only for the subdomain of this rank.

    Returns
    -------
    RunResult
        Only this rank's subdomain (lists of length 1),
        use gather_result to collect all subdomains on rank 0.
    """
    rank = comm.Get_rank()
    left_nb, right_nb = neighbours(comm)
    has_left = use_neighbours and left_nb != MPI.PROC_NULL
    has_right = use_neighbours and right_nb != MPI.PROC_NULL

    # Receive buffers, allocated once and filled in every step.
    # Only used at an interface, None in computeF_b_edges gives tanh(t)
    from_left = np.empty(grid.Nmu)
    from_right = np.empty(grid.Nmu)
    timer = PhaseTimer()
    t = 0.0
    step = 0
    time = [t]
    rank_hist = [grid.r]
    snap_t, snap_f = [], []

    if snap_every:
        snap_t.append(t)
        snap_f.append(full_f(lr))

    t_start = perf_counter()
    with tqdm(total=t_f / dt, desc="Running Simulation",
              disable=not progress or rank != 0) as pbar:
        while t < t_f:
            pbar.update(1)

            if t + dt > t_f:
                dt = t_f - t

            # 1) Boundary values from the neighbours' state before the step
            if use_neighbours:
                with timer("exchange"):
                    exchange_edges(comm, lr, left_nb, right_nb,
                                   from_left, from_right)

            with timer("F_b"):
                F_b = computeF_b_edges(
                    t, lr, grid,
                    inflow_left=from_left if has_left else None,
                    inflow_right=from_right if has_right else None,
                )

            # 2) Local update, same as in the serial version
            lr, grid = local_step(lr, grid, F_b, t, dt, rng,
                                  tol_sing_val, drop_tol, timer, option)

            t += dt
            step += 1
            time.append(t)
            rank_hist.append(grid.r)

            if snap_every and (step % snap_every == 0 or t >= t_f):
                snap_t.append(t)
                snap_f.append(full_f(lr))

    return RunResult(
        lrs=[lr],
        grids=[grid],
        time=time,
        rank_hist=[rank_hist],
        timings=[timer.times],
        mem_peak=[timer.mem_peak],
        snap_t=snap_t,
        snap_f=snap_f,
        runtime=perf_counter() - t_start,
    )

def gather_result(comm, local: RunResult, root: int=0) -> RunResult | None:
    """ 
    Collect the results of ranks on root

    Returns the RunResult of all subdomains on root
    """
    parts = comm.gather(local, root=root)
    if comm.Get_rank() != root:
        return None
    
    snap_f = [np.concatenate([p.snap_f[k] for p in parts], axis=0)
              for k in range(len(local.snap_f))]

    return RunResult(
        lrs=[lr for p in parts for lr in p.lrs],
        grids=[g for p in parts for g in p.grids],
        time=local.time,  # same on every rank
        rank_hist=[h for p in parts for h in p.rank_hist],
        timings=[tim for p in parts for tim in p.timings],
        mem_peak=[m for p in parts for m in p.mem_peak],
        snap_t=local.snap_t,
        snap_f=snap_f,
        runtime=max(p.runtime for p in parts),
    )

if __name__ == "__main__":
    # Check against integrate_serial
    from DLR_rt.src.dd_1x1d import integrate_serial, setup_domains

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    print(f"Rank {rank} of {size} starting")

    Nx, Nmu, r, t_f, dt, tol, seed = 256, 64, 5, 0.2, 1e-3, 1e-5, 0

    grids, lr0s, rngs = setup_domains(Nx, Nmu, r,size, seed)
    comm.Barrier()
    local = integrate_mpi(lr0s[rank], grids[rank], rngs[rank], t_f, dt, comm,
                          tol, tol, snap_every=10)
    res = gather_result(comm, local)

    if rank == 0:
        grids, lr0s, rngs = setup_domains(Nx, Nmu, r, size, seed)
        ser = integrate_serial(lr0s, grids, rngs, t_f, dt, tol, tol, snap_every=10)

        identical = np.array_equal(res.f(), ser.f())
        same_snaps = all(np.array_equal(a, b) for a, b in zip(res.snap_f, ser.snap_f))
        print(f"\n{size} ranks vs serial with {size} domains:")
        print(f"  bit-identical f:         {identical}")
        print(f"  bit-identical snapshots: {same_snaps}")
        print(f"  same rank history:       {res.rank_hist == ser.rank_hist}")
        print(f"  final ranks:             {[h[-1] for h in res.rank_hist]}")

        speedup = ser.runtime / res.runtime
        print(f"  serial {ser.runtime:.2f} s, MPI {res.runtime:.2f} s, "
              f"speedup {speedup:.2f}, efficiency {speedup / size * 100:.0f} %")
        for i, tim in enumerate(res.timings):
            print(f"  rank {i}: " + ", ".join(f"{k} {v:.2f} s" for k, v in tim.items()))

        assert identical and same_snaps
        assert res.rank_hist == ser.rank_hist
        print("All checks passed")
