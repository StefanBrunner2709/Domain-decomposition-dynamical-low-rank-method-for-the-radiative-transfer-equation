"""
Domain decomposition for the 1x1d radiative transfer equation, MPI version.

Every rank gets a contiguous block of subdomains (n_domains >= number of ranks).
Inside a block the step is the same as serial (block_step), between ranks
only one row of length Nmu per side is exchanged.

Run: mpirun -n 4 python -m DLR_rt.src.dd_mpi 8   (8 domains on 4 ranks)
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

from DLR_rt.src.dd_1x1d import PhaseTimer, RunResult, block_step, full_f
from DLR_rt.src.grid import Grid_1x1d
from DLR_rt.src.lr import LR, first_row, last_row


def neighbours(comm):
    """Left and right neighbour rank, PROC_NULL at the physical boundary."""
    rank, size = comm.Get_rank(), comm.Get_size()
    left_nb = rank - 1 if rank > 0 else MPI.PROC_NULL
    right_nb = rank + 1 if rank < size - 1 else MPI.PROC_NULL
    return left_nb, right_nb


def domain_slice(n_domains: int, comm) -> slice:
    """Block of subdomains of this rank, the first ranks get one more if uneven."""
    rank, size = comm.Get_rank(), comm.Get_size()
    assert n_domains >= size, "Need at least one subdomain per rank"
    base, extra = divmod(n_domains, size)
    start = rank * base + min(rank, extra)
    return slice(start, start + base + (rank < extra))


def exchange_edges(comm, lrs: list[LR], left_nb, right_nb,
                   from_left: np.ndarray, from_right: np.ndarray):
    """Send the outer rows of the block to the neighbours, receive theirs."""
    comm.Sendrecv(last_row(lrs[-1]), dest=right_nb, recvbuf=from_left, source=left_nb)
    comm.Sendrecv(first_row(lrs[0]), dest=left_nb, recvbuf=from_right, source=right_nb)


def integrate_mpi(lrs: list[LR], grids: list[Grid_1x1d], rngs: list,
                  t_f: float, dt: float, comm=MPI.COMM_WORLD,
                  tol_sing_val: float = 1e-6, drop_tol: float = 1e-6,
                  option: str = "lie", use_neighbours: bool = True,
                  snap_every: int | None = None,
                  progress: bool = True) -> RunResult:
    """
    Integrate this rank's block of subdomains, all ranks call this together.

    Same parameters as dd_1x1d.integrate_serial. Returns only this rank's
    subdomains, use gather_result to collect all of them on rank 0.
    """
    rank = comm.Get_rank()
    n = len(lrs)
    lrs, grids = list(lrs), list(grids)
    left_nb, right_nb = neighbours(comm)
    has_left = use_neighbours and left_nb != MPI.PROC_NULL
    has_right = use_neighbours and right_nb != MPI.PROC_NULL
    from_left = np.empty(grids[0].Nmu)
    from_right = np.empty(grids[-1].Nmu)

    timers = [PhaseTimer() for _ in range(n)]
    rank_timer = PhaseTimer()
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
    with tqdm(total=t_f / dt, desc="Running Simulation",
              disable=not progress or rank != 0) as pbar:
        while t < t_f:
            pbar.update(1)

            if t + dt > t_f:
                dt = t_f - t

            if use_neighbours:
                with rank_timer("exchange"):
                    exchange_edges(comm, lrs, left_nb, right_nb, from_left, from_right)

            block_step(lrs, grids, rngs, timers, t, dt, tol_sing_val, drop_tol,
                       option, use_neighbours,
                       outer_left=from_left if has_left else None,
                       outer_right=from_right if has_right else None)

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
        owner=[rank] * n,
        rank_timings=[rank_timer.times],
    )


def gather_result(comm, local: RunResult, root: int = 0) -> RunResult | None:
    """Collect all subdomains on root (left to right), None on the other ranks."""
    parts = comm.gather(local, root=root)
    if comm.Get_rank() != root:
        return None

    def join(attr):
        return [x for p in parts for x in getattr(p, attr)]

    return RunResult(
        lrs=join("lrs"),
        grids=join("grids"),
        time=local.time,
        rank_hist=join("rank_hist"),
        timings=join("timings"),
        mem_peak=join("mem_peak"),
        snap_t=local.snap_t,
        snap_f=[np.concatenate([p.snap_f[k] for p in parts], axis=0)
                for k in range(len(local.snap_f))],
        runtime=max(p.runtime for p in parts),  # slowest rank
        owner=join("owner"),
        rank_timings=join("rank_timings"),
    )


def timings_per_rank(res: RunResult) -> list[dict]:
    """Timings summed over the domains of each rank, plus its exchange."""
    per_rank = [dict(tim) for tim in res.rank_timings]
    for owner, tim in zip(res.owner, res.timings):
        for phase, seconds in tim.items():
            per_rank[owner][phase] = per_rank[owner].get(phase, 0.0) + seconds
    return per_rank


if __name__ == "__main__":
    # Check against integrate_serial, used by test_mpi_matches_serial.
    # Run: mpirun -n 4 python -m DLR_rt.src.dd_mpi 8   (8 domains, default: 1 per rank)
    import sys

    from DLR_rt.src.dd_1x1d import integrate_serial, setup_domains

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    n_domains = int(sys.argv[1]) if len(sys.argv) > 1 else size
    setup = dict(Nx=256, Nmu=64, r=5, n_domains=n_domains, seed=0)
    run = dict(t_f=0.2, dt=1e-3, tol_sing_val=1e-5, drop_tol=1e-5, snap_every=10)

    grids, lr0s, rngs = setup_domains(**setup)
    mine = domain_slice(n_domains, comm)
    comm.Barrier()
    local = integrate_mpi(lr0s[mine], grids[mine], rngs[mine], comm=comm, **run)
    res = gather_result(comm, local)

    if rank == 0:
        grids, lr0s, rngs = setup_domains(**setup)
        ser = integrate_serial(lr0s, grids, rngs, **run)

        identical = (np.array_equal(res.f(), ser.f())
                     and len(res.snap_f) == len(ser.snap_f)
                     and all(map(np.array_equal, res.snap_f, ser.snap_f))
                     and res.rank_hist == ser.rank_hist)
        print(f"\n{size} ranks, {n_domains} domains "
              f"{[res.owner.count(i) for i in range(size)]}, "
              f"bit-identical to serial: {identical}, "
              f"speedup {ser.runtime / res.runtime:.2f}")
        for i, tim in enumerate(timings_per_rank(res)):
            print(f"  rank {i}: " + ", ".join(f"{k} {v:.2f} s" for k, v in tim.items()))

        assert identical
        print("All checks passed")
