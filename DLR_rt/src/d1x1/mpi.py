"""
MPI version of the 1x1d domain decomposition.

Every rank gets a contiguous block of subdomains (n_domains >= number of ranks).
Inside a block the step is the same as serial (domains.block_step), between ranks
only one row of length Nmu per side is exchanged.

    grids, lrs, rngs = setup_domains(Nx, Nmu, r, n_domains)   # on every rank
    mine = slice_of(even_bounds(n_domains, comm.Get_size()), comm.Get_rank())
    local = integrate_mpi(lrs[mine], grids[mine], rngs[mine], t_f, dt, comm=comm)
    res = gather_result(comm, local)                          # rank 0, else None

Self-check against serial (used by the tests):
    mpirun -n 4 python -m DLR_rt.src.d1x1.mpi 8            even split, 8 subdomains
    mpirun -n 3 python -m DLR_rt.src.d1x1.mpi 8 brute      measure, then optimal split
    mpirun -n 3 python -m DLR_rt.src.d1x1.mpi 8 balanced   the same with the bisection
    mpirun -n 3 python -m DLR_rt.src.d1x1.mpi 12 dynamic   rebalance every 5 steps
"""
import os

# BLAS and LAPACK to 1 thread per MPI rank, has to be set before numpy is imported
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from time import perf_counter  # noqa: E402

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402
from tqdm import tqdm  # noqa: E402

from DLR_rt.src.d1x1.domains import PhaseTimer, RunResult, block_step  # noqa: E402
from DLR_rt.src.d1x1.inflow import tanh  # noqa: E402
from DLR_rt.src.d1x1.lowrank import LR  # noqa: E402
from DLR_rt.src.grid import Grid_1x1d  # noqa: E402


def neighbours(comm):
    """Left and right neighbour rank, PROC_NULL at the physical boundary."""
    rank, size = comm.Get_rank(), comm.Get_size()
    left_nb = rank - 1 if rank > 0 else MPI.PROC_NULL
    right_nb = rank + 1 if rank < size - 1 else MPI.PROC_NULL
    return left_nb, right_nb


def exchange_edges(comm, lrs: list[LR], left_nb, right_nb,
                   from_left: np.ndarray, from_right: np.ndarray):
    """Send the outer rows of the block to the neighbours, receive theirs."""
    comm.Sendrecv(lrs[-1].last_row(), dest=right_nb, recvbuf=from_left, source=left_nb)
    comm.Sendrecv(lrs[0].first_row(), dest=left_nb, recvbuf=from_right,
                  source=right_nb)


def integrate_mpi(lrs: list[LR], grids: list[Grid_1x1d], rngs: list,
                  t_f: float, dt: float, comm=MPI.COMM_WORLD,
                  tol_sing_val: float = 1e-6, drop_tol: float = 1e-6, inflow=tanh,
                  use_neighbours: bool = True, snap_every: int | None = None,
                  progress: bool = True, balancer=None) -> RunResult:
    """
    Integrate this rank's block of subdomains, all ranks call this together.

    Same parameters as domains.integrate_serial. Returns only this rank's
    subdomains, gather_result collects all of them on rank 0.

    balancer: balancer.Balancer for dynamic load balancing. Every balancer.every
    steps (before the exchange) the subdomains are redistributed, together with
    their rng, timers and rank history, so the result stays bit-identical.
    """
    rank = comm.Get_rank()
    lrs, grids, rngs = list(lrs), list(grids), list(rngs)
    left_nb, right_nb = neighbours(comm)
    has_left = use_neighbours and left_nb != MPI.PROC_NULL
    has_right = use_neighbours and right_nb != MPI.PROC_NULL
    from_left = np.empty(grids[0].Nmu)
    from_right = np.empty(grids[-1].Nmu)

    timers = [PhaseTimer() for _ in lrs]
    rank_timer = PhaseTimer()  # exchange and rebalance of this rank
    rank_busy = 0.0  # time in block_step of this rank, whatever subdomains it has
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
    with tqdm(total=t_f / dt, desc="Running Simulation",
              disable=not progress or rank != 0) as pbar:
        while t < t_f:
            pbar.update(1)
            if t + dt > t_f:
                dt = t_f - t

            if balancer is not None:
                # Every rank keeps a non-empty contiguous block, so the neighbour
                # ranks and the exchange buffers (length Nmu) stay the same
                with rank_timer("rebalance"):
                    d = balancer.step(step, t, dict(lrs=lrs, grids=grids, rngs=rngs,
                                                    timers=timers, rank_hist=rank_hist))
                lrs, grids, rngs = d["lrs"], d["grids"], d["rngs"]
                timers, rank_hist = d["timers"], d["rank_hist"]

            if use_neighbours:
                with rank_timer("exchange"):
                    exchange_edges(comm, lrs, left_nb, right_nb, from_left, from_right)

            t0 = perf_counter()
            block_step(lrs, grids, rngs, timers, t, dt, tol_sing_val, drop_tol,
                       inflow, use_neighbours,
                       outer_left=from_left if has_left else None,
                       outer_right=from_right if has_right else None)
            rank_busy += perf_counter() - t0

            t += dt
            step += 1
            time.append(t)
            for i, lr in enumerate(lrs):
                rank_hist[i].append(lr.r)
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
        owner=[rank] * len(lrs),
        rank_timings=[rank_timer.times],
        rank_busy=[rank_busy],
        balance=balancer.summary() if balancer is not None else {},
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
        rank_busy=join("rank_busy"),
        balance=local.balance,  # the same on every rank
    )


if __name__ == "__main__":
    # brute / balanced: first an even run, its busy time per subdomain gives the
    # weights, then the same run with the balanced split. dynamic: a deliberately bad
    # start that the balancer has to repair. All have to be bit-identical to serial,
    # only the distribution of the subdomains changes.
    import sys

    from DLR_rt.src.d1x1.balancer import Balancer
    from DLR_rt.src.d1x1.domains import integrate_serial, setup_domains
    from DLR_rt.src.d1x1.inflow import beam
    from DLR_rt.src.d1x1.partition import (
        balanced_bounds,
        balanced_bounds_brute,
        even_bounds,
        measured_weights,
        slice_of,
    )

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    n_domains = int(sys.argv[1]) if len(sys.argv) > 1 else size
    mode = sys.argv[2] if len(sys.argv) > 2 else "even"
    assert mode in ("even", "brute", "balanced", "dynamic"), f"Unknown mode {mode}"
    setup = dict(Nx=256, Nmu=64, r=5, n_domains=n_domains, seed=0)
    run = dict(t_f=0.2, dt=1e-3, tol_sing_val=1e-5, drop_tol=1e-5, snap_every=10)
    if mode != "even":
        # One-sided beam: the left subdomains need a higher rank, the cost differs
        run["inflow"] = beam

    def run_mpi(bounds, balancer=None):
        """All ranks together with the split bounds, result on rank 0."""
        grids, lrs, rngs = setup_domains(**setup)
        mine = slice_of(bounds, rank)
        comm.Barrier()
        local = integrate_mpi(lrs[mine], grids[mine], rngs[mine], comm=comm,
                              progress=False, balancer=balancer, **run)
        return gather_result(comm, local)

    results = {"even": run_mpi(even_bounds(n_domains, size))}
    if mode == "dynamic":
        # One subdomain per rank, the last gets the rest
        start = [*range(size), n_domains]
        bal = Balancer(comm, start, every=5, weights="measured")
        results["dynamic"] = run_mpi(start, bal)
        if rank == 0:
            print(f"dynamic: {bal.summary()['n_changes']} changes, {bal.n_moved} moved, "
                  f"final bounds {bal.bounds}")
    elif mode != "even":
        # Weights from the even run, computed on rank 0 and the same on every rank
        weights = comm.bcast(measured_weights(results["even"].timings)
                             if rank == 0 else None, root=0)
        split = balanced_bounds_brute if mode == "brute" else balanced_bounds
        results[mode] = run_mpi(split(weights, size))

    if rank == 0:
        grids, lrs, rngs = setup_domains(**setup)
        ser = integrate_serial(lrs, grids, rngs, progress=False, **run)

        all_identical = True
        for name, res in results.items():
            identical = (np.array_equal(res.f(), ser.f())
                         and len(res.snap_f) == len(ser.snap_f)
                         and all(map(np.array_equal, res.snap_f, ser.snap_f))
                         and res.rank_hist == ser.rank_hist)
            all_identical = all_identical and identical
            busy = res.rank_busy  # per rank, correct also if subdomains migrated
            print(f"\n{name}: {size} ranks, {n_domains} domains "
                  f"{[res.owner.count(i) for i in range(size)]}, "
                  f"bit-identical to serial: {identical}, "
                  f"speedup vs serial {ser.runtime / res.runtime:.2f}")
            print(f"  busy per rank {[round(b, 3) for b in busy]} s, "
                  f"max / mean {max(busy) / np.mean(busy):.2f}")
        if mode != "even":
            print(f"\n{mode} split vs even split: "
                  f"{results['even'].runtime / results[mode].runtime:.2f}x faster")

        assert all_identical
        print("All checks passed")
