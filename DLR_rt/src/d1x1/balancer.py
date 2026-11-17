"""
Dynamic load balancing of the MPI ranks for the 1x1d domain decomposition.

Every `every` steps (top of the time loop in mpi.integrate_mpi, before the edge
exchange) the weights of all subdomains are gathered, a new contiguous split is
computed with partition.balanced_bounds and the subdomains are migrated to their
new owners.

Only the owner of a subdomain changes, not its geometry. Every rank keeps a
contiguous, non-empty block, so the neighbours stay rank-1 and rank+1. The random
generator moves with its subdomain, so the result stays bit-identical to serial.

Usage:
    bal = Balancer.from_local(comm, n_local, every=10, weights="model")
    local = integrate_mpi(..., balancer=bal)
"""
from time import perf_counter

import numpy as np
from mpi4py import MPI

from DLR_rt.src.d1x1.partition import balanced_bounds, max_block, model_weights, owners


def migrate(comm, old: list[int], new: list[int], domains: dict) -> dict:
    """
    Send every local subdomain to its owner in the new split.

    domains: dict of lists of the local subdomains (left to right), all of the
    same length. Returns the same keys with the local subdomains of the new split.
    Pickled objects with alltoall, usually only rank-1 and rank+1 get something.
    """
    rank, p = comm.Get_rank(), comm.Get_size()
    keys = list(domains)
    send = [[] for _ in range(p)]
    for j, g in enumerate(range(old[rank], old[rank + 1])):
        dest = int(np.searchsorted(new, g, side="right")) - 1
        send[dest].append((g, [domains[k][j] for k in keys]))

    items = sorted((x for part in comm.alltoall(send) for x in part),
                   key=lambda x: x[0])
    assert [g for g, _ in items] == list(range(new[rank], new[rank + 1])), \
        "Migration lost or duplicated a subdomain"
    return {k: [vals[i] for _, vals in items] for i, k in enumerate(keys)}


class Balancer:
    """
    Dynamic repartitioning of the contiguous blocks of subdomains every `every` steps.

    Parameters
    ----------
    comm: MPI communicator
    bounds : list[int]
        Current split, rank k owns bounds[k]:bounds[k+1]. Same on every rank.
    every : int
        Repartition every `every` steps.
    weights: "model" or "measured"
        "model": (Nx + Nmu) r^2 + r^3 from the current ranks of the subdomains,
        "measured": the busy time per subdomain since the last rebalance.
    min_gain : float
        Hysteresis: only migrate if the new split lowers the largest block cost by
        at least this fraction (0.05 = 5 %). Without it the noise of the measured
        weights makes the split flip back and forth between nearly equal splits.
    """

    def __init__(self, comm, bounds, every: int = 10, weights: str = "measured",
                 min_gain: float = 0.05):
        assert weights in ("model", "measured"), f"Unknown weights {weights}"
        assert len(bounds) == comm.Get_size() + 1, "bounds must be of length p+1"
        assert every >= 1, "every must be at least 1"
        self.comm = comm
        self.bounds = [int(b) for b in bounds]  # python ints, numpy ints are not JSON
        self.every = every
        self.weights = weights
        self.min_gain = min_gain
        self._busy0 = None  # busy time per local subdomain at the last rebalance
        self.history = [(0.0, list(self.bounds))]  # (t, bounds) after every change
        self.events = []  # (step, moved, cost) of every rebalance

    @classmethod
    def from_local(cls, comm, n_local: int, **kw):
        """Builds the bounds from the number of local subdomains on every rank."""
        counts = comm.allgather(n_local)
        return cls(comm, [0, *np.cumsum(counts)], **kw)

    @property
    def n_moved(self) -> int:
        return sum(moved for _, moved, _ in self.events)

    def summary(self) -> dict:
        """Everything for the benchmark JSON (the same on every rank)."""
        return dict(every=self.every, weights=self.weights, min_gain=self.min_gain,
                    n_rebalances=len(self.events),
                    n_changes=len(self.history) - 1, n_moved=self.n_moved,
                    cost=float(sum(c for _, _, c in self.events)),
                    history=self.history)

    def _busy(self, timers) -> np.ndarray:
        """Busy time of every local subdomain since the last rebalance."""
        now = np.array([sum(tm.times.values()) for tm in timers])
        return now - self._busy0 if self._busy0 is not None else now

    def step(self, step: int, t: float, domains: dict) -> dict:
        """
        Call at the top of the time loop, collectively on all ranks.

        domains: dict "lrs", "grids", "rngs", "timers", "rank_hist".
        Returns the same dict, or a new one if subdomains were migrated.
        """
        if step == 0 or step % self.every:
            return domains

        t0 = perf_counter()
        if self.weights == "model":  # current rank = last entry of the history
            local = model_weights(domains["grids"],
                                  [[h[-1]] for h in domains["rank_hist"]])
        else:
            local = self._busy(domains["timers"])
        w = np.concatenate(self.comm.allgather(np.asarray(local, dtype=float)))
        new = balanced_bounds(w, self.comm.Get_size())
        if max_block(w, new) > (1.0 - self.min_gain) * max_block(w, self.bounds):
            new = self.bounds  # gain too small for a migration (hysteresis)

        moved = 0
        if new != self.bounds:
            moved = int(np.sum(owners(self.bounds) != owners(new)))
            domains = migrate(self.comm, self.bounds, new, domains)
            self.bounds = new
            self.history.append((t, list(new)))

        self._busy0 = np.array([sum(tm.times.values()) for tm in domains["timers"]])
        cost = self.comm.allreduce(perf_counter() - t0, op=MPI.MAX)
        self.events.append((step, moved, cost))
        return domains
