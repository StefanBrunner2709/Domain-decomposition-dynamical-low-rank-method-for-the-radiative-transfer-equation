"""
Contiguous splits of n subdomains onto p MPI ranks, no MPI in here.

A split is given by its bounds of length p + 1: rank k owns the subdomains
bounds[k]:bounds[k+1], bounds[0] = 0, bounds[-1] = n, every rank at least one.
The weights are the cost of every subdomain, from the cost model or measured.

    bounds = balanced_bounds(measured_weights(res.timings), p)
    mine = slice_of(bounds, rank)
"""
import itertools
import math

import numpy as np


### Splits

def even_bounds(n: int, p: int) -> list[int]:
    """Same number of subdomains per rank, the first n % p ranks get one more."""
    assert n >= p, "Need at least one subdomain per rank"
    base, extra = divmod(n, p)
    return [k * base + min(k, extra) for k in range(p + 1)]


def slice_of(bounds: list[int], rank: int) -> slice:
    """The subdomains of rank."""
    return slice(bounds[rank], bounds[rank + 1])


def owners(bounds: list[int]) -> np.ndarray:
    """Owner rank of every subdomain 0..n-1."""
    return np.searchsorted(bounds, np.arange(bounds[-1]), side="right") - 1


### Cost of a split

def block_costs(weights, bounds: list[int]) -> np.ndarray:
    """Summed weight of every block, i.e. the cost of every rank."""
    w = np.asarray(weights, dtype=float)
    return np.array([w[a:b].sum() for a, b in zip(bounds, bounds[1:])])


def max_block(weights, bounds: list[int]) -> float:
    """Largest block, the cost of the slowest rank."""
    return float(block_costs(weights, bounds).max())


def block_imbalance(weights, bounds: list[int]) -> float:
    """Max / mean of the blocks, 1 = perfectly balanced."""
    blocks = block_costs(weights, bounds)
    return float(blocks.max() / blocks.mean())


### Balanced splits: smallest maximum block

def prefix_sum(weights) -> np.ndarray:
    """[0, w0, w0 + w1, ...], the sum of a block a:b is prefix[b] - prefix[a]."""
    return np.concatenate(([0.0], np.cumsum(weights)))


def _probe(prefix: np.ndarray, p: int, bottleneck: float) -> list[int] | None:
    """
    Greedy split where no block is larger than bottleneck, None if there is none.

    Every rank takes as many subdomains as fit, but leaves at least one for each
    of the remaining ranks. Example w = [1, 1, 1, 1, 50, 60, 1, 1, 1, 1], p = 4:
    bottleneck 60 gives [0, 5, 6, 9, 10] (blocks 54, 60, 3, 1), 59 gives None.
    """
    n = len(prefix) - 1
    bounds = [0]
    for k in range(p):
        start = bounds[-1]
        if k == p - 1:
            stop = n
        else:
            # furthest stop with prefix[stop] - prefix[start] <= bottleneck
            stop = int(np.searchsorted(prefix, prefix[start] + bottleneck,
                                       side="right")) - 1
            stop = min(stop, n - (p - k - 1))
        if stop <= start or prefix[stop] > prefix[start] + bottleneck:
            return None
        bounds.append(stop)
    return bounds


def balanced_bounds(weights, p: int, rtol: float = 1e-10) -> list[int]:
    """
    Split with the smallest maximum block, by bisection over the bottleneck.

    The optimal bottleneck lies between max(sum / p, max w) and sum, _probe says
    if a bottleneck is feasible. O(p log(n) log(1/rtol)), also for large n and p.
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
            lo = mid
        else:
            hi, best = mid, cand
    return best


def balanced_bounds_brute(weights, p: int, max_splits: int = 2_000_000) -> list[int]:
    """
    Try all (n-1 choose p-1) splits, take the one with the smallest maximum block.

    455 splits for n = 16, p = 4, but 2.6 million for n = 32, p = 8:
    only for small systems and as reference for balanced_bounds.
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


### Weights

def model_weights(grids, rank_hist: list[list[int]]) -> np.ndarray:
    """
    Cost model of one step per subdomain, (Nx + Nmu) r^2 + r^3, averaged over time.

    rank_hist: [domain][step] low rank r, e.g. RunResult.rank_hist, or
    [[lr.r] for lr in lrs] for the current ranks only.
    The fixed cost per subdomain (Python, exchange) is not in the model.
    """
    return np.array([np.mean((g.Nx + g.Nmu) * np.asarray(h, float)**2
                             + np.asarray(h, float)**3)
                     for g, h in zip(grids, rank_hist)])


def measured_weights(timings: list[dict]) -> np.ndarray:
    """
    Busy time of every subdomain, e.g. RunResult.timings, without the exchange.

    Contains the real ranks over time and the fixed cost per subdomain.
    """
    return np.array([sum(tim.values()) for tim in timings])
