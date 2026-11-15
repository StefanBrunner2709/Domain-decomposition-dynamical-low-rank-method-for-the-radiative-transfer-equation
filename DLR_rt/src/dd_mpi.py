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

import itertools
import math

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


"""
Splitting method for 1D
"""
def model_weights(grids: list[Grid_1x1d]) -> np.ndarray:
    """ 
    Uses a cost model for one step per subdomain
    here i kind of adapted this from the paper (Nx + Nmu) * r² + r³ 
    """
    return np.array([(g.Nx + g.Nmu) * g.r**2 + g.r**3 for g in grids])

def prefix_sum(weights: np.ndarray) -> np.ndarray:
    """ 
    Computes the cummulative sum so any blocks cost is just one substraction
    """
    return np.concatenate(([0.0], np.cumsum(weights)))

def measured_weights(res: RunResult) -> np.ndarray:
    """
    Busy time of every subdomain in a previous run (all phases, without exchange).

    Better weights than model_weights: contains the real ranks over time and the
    fixed cost per subdomain (~1.3 ms per step, see benchmark_serial domains).
    """
    return np.array([sum(tim.values()) for tim in res.timings])


def balanced_bounds_brute(weights: np.ndarray, p: int,
                          max_splits: int = 2_000_000) -> list[int]:
    """
    Try all contiguous splits and take the one with the smallest maximum block sum.

    A split of n subdomains onto p ranks is fixed by p-1 cut positions out of the
    n-1 gaps, so there are (n-1 choose p-1) splits: 455 for n=16, p=4, but
    2.6 million for n=32, p=8. Only for small systems and as reference in tests.
    """
    w = np.asarray(weights, dtype=float)
    n = len(w)
    assert n >= p, "Need at least one subdomain per rank"
    n_splits = math.comb(n - 1, p - 1)
    assert n_splits <= max_splits, (f"{n_splits} splits are too many for brute force, "
                                    f"use balanced_bounds")
    prefix = prefix_sum(w)
    best, best_cost = None, np.inf
    for cuts in itertools.combinations(range(1, n), p - 1):
        b = (0, *cuts, n)
        cost = max(prefix[b[i + 1]] - prefix[b[i]] for i in range(p))
        if cost < best_cost:
            best, best_cost = list(b), cost
    return best

def _probe(prefix: np.ndarray, p:int, bottleneck: float) -> list[int] | None:
    """ 
    Idea is to cut such that no rank has more than bottleneck amout of work

    Example B = 60, w = [1,1,1,1,50, 60, 1,1,1,1] p=4
    rank 0: 1,1,1,1,50 = 54
    rank 1: 60
    rank 2: 1+1+1 = 3 # must leave at least one for rank 3
    rank 3: 1

    Then B=59 doesnt work, because rank 1 has 60 > 59. So we return None
    """
    n = len(prefix) - 1 
    bounds = [0]
    for k in range(p):
        start = bounds[-1]
        if k == p - 1:
            stop = n
        else:
            # furthest stop with prefix[stop] - prefix[start] <= bottleneck
            stop = int(np.searchsorted(prefix, prefix[start] + bottleneck, side="right")) - 1
            stop = min(stop, n - (p - k - 1))  # leave at least one for each remaining rank
        if stop <= start or prefix[stop] > prefix[start] + bottleneck:
            return None  # bottleneck B is too small
        bounds.append(stop)
    return bounds

def balanced_bounds(weights: np.ndarray, p: int, rtol: float = 1e-10) -> list[int]:
    """
    Contiguous split of the weights into p blocks with the smallest maximum block sum.

    Bisection over the bottleneck B: _probe(B) says if a split with max block <= B
    exists. Returns the bounds, rank k gets the domains bounds[k]:bounds[k+1].
    """
    w = np.asarray(weights, dtype=float)
    assert len(w) >= p, "Need at least one subdomain per rank"
    prefix = prefix_sum(w)
    lo = max(prefix[-1] / p, w.max())  # no split can be better than this
    hi = prefix[-1]                    # all work on one rank, always feasible
    best = _probe(prefix, p, lo)
    if best is not None:               # the lower bound is reached already
        return best
    best = _probe(prefix, p, hi)
    while hi - lo > rtol * hi:
        mid = 0.5 * (lo + hi)
        cand = _probe(prefix, p, mid)
        if cand is None:
            lo = mid                   # too small, a larger bottleneck is needed
        else:
            hi, best = mid, cand       # feasible, try a smaller one
    return best


def _slice_of_rank(bounds: list[int], comm) -> slice:
    """Rank k gets the subdomains bounds[k]:bounds[k+1]."""
    rank = comm.Get_rank()
    return slice(bounds[rank], bounds[rank + 1])


def domain_slice_brute(weights, comm) -> slice:
    """
    Contiguous block of this rank from the optimal split, found by brute force.

    Only for small systems. The weights have to be the same on every rank
    (compute on rank 0 and bcast), then every rank finds the same split.
    """
    return _slice_of_rank(balanced_bounds_brute(weights, comm.Get_size()), comm)


def domain_slice_balanced(weights, comm) -> slice:
    """Like domain_slice_brute, but with the bisection, also for large systems."""
    return _slice_of_rank(balanced_bounds(weights, comm.Get_size()), comm)


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
                  progress: bool = True, inflow=None) -> RunResult:
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
                       outer_right=from_right if has_right else None,
                       inflow=inflow)

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


if __name__ == "__main__":
    # Check against integrate_serial, used by the tests:
    #   mpirun -n 4 python -m DLR_rt.src.dd_mpi 8           even split, 8 domains
    #   mpirun -n 3 python -m DLR_rt.src.dd_mpi 8 brute     measure, then optimal split
    #   mpirun -n 3 python -m DLR_rt.src.dd_mpi 8 balanced  the same with the bisection
    # brute / balanced: first an even run, its busy time per subdomain gives the
    # weights, then the same run with the balanced split. Both have to be
    # bit-identical to serial, only the distribution of the subdomains changes.
    import sys

    from DLR_rt.src.dd_1x1d import beam_inflow, integrate_serial, setup_domains

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    n_domains = int(sys.argv[1]) if len(sys.argv) > 1 else size
    mode = sys.argv[2] if len(sys.argv) > 2 else "even"
    assert mode in ("even", "brute", "balanced"), f"Unknown mode {mode}"
    setup = dict(Nx=256, Nmu=64, r=5, n_domains=n_domains, seed=0)
    run = dict(t_f=0.2, dt=1e-3, tol_sing_val=1e-5, drop_tol=1e-5, snap_every=10)
    if mode != "even":
        # One-sided beam: the left subdomains need a higher rank, the cost differs
        run["inflow"] = beam_inflow

    def run_mpi(get_slice):
        """All ranks together with the given distribution, result on rank 0."""
        grids, lr0s, rngs = setup_domains(**setup)
        mine = get_slice()
        comm.Barrier()
        local = integrate_mpi(lr0s[mine], grids[mine], rngs[mine], comm=comm,
                              progress=False, **run)
        return gather_result(comm, local)

    results = {"even": run_mpi(lambda: domain_slice(n_domains, comm))}
    if mode != "even":
        # Weights from the even run, computed on rank 0 and the same on every rank
        weights = comm.bcast(measured_weights(results["even"]) if rank == 0 else None,
                             root=0)
        split = domain_slice_brute if mode == "brute" else domain_slice_balanced
        results[mode] = run_mpi(lambda: split(weights, comm))

    if rank == 0:
        grids, lr0s, rngs = setup_domains(**setup)
        ser = integrate_serial(lr0s, grids, rngs, progress=False, **run)

        all_identical = True
        for name, res in results.items():
            identical = (np.array_equal(res.f(), ser.f())
                         and len(res.snap_f) == len(ser.snap_f)
                         and all(map(np.array_equal, res.snap_f, ser.snap_f))
                         and res.rank_hist == ser.rank_hist)
            all_identical = all_identical and identical
            busy = [sum(s for k, s in tim.items() if k != "exchange")
                    for tim in res.timings_per_rank()]
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
