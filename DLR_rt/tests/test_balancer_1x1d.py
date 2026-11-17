"""
Tests for DLR_rt/src/d1x1/balancer.py.

The collective calls (allgather, alltoall, allreduce) run on fake ranks in threads,
without mpirun. The full run with migrations is in test_mpi_1x1d (dynamic).
"""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from DLR_rt.src.d1x1.balancer import Balancer, migrate  # noqa: E402


class ThreadComm:
    """Fake communicator for p ranks, every rank runs in its own thread."""

    def __init__(self, rank, size, shared):
        self.rank, self.size, self.shared = rank, size, shared

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def _exchange(self, obj):
        """Every rank puts obj, every rank gets the list of all p."""
        sh = self.shared
        sh["buf"][self.rank] = obj
        sh["barrier"].wait()
        out = list(sh["buf"])
        sh["barrier"].wait()
        return out

    def allgather(self, obj):
        return self._exchange(obj)

    def alltoall(self, send):
        """Rank j receives send[j] of every rank."""
        return [part[self.rank] for part in self._exchange(send)]

    def allreduce(self, x, op=None):
        vals = self._exchange(x)
        return max(vals) if op == MPI.MAX else sum(vals)


def run_ranks(p, fn):
    """Run fn(comm) on p fake ranks in parallel, returns the results by rank."""
    shared = dict(buf=[None] * p, barrier=threading.Barrier(p, timeout=10))
    results, errors = [None] * p, []

    def worker(rank):
        try:
            results[rank] = fn(ThreadComm(rank, p, shared))
        except BaseException as e:
            errors.append(e)
            shared["barrier"].abort()  # do not let the other ranks wait forever

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(p)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    return results


def fake_domains(bounds, rank, ranks_r, Nx=64, Nmu=64, busy=None):
    """
    Local subdomains of rank for the split bounds, without a real simulation.

    Every subdomain g is labelled with its global index, so after a migration we
    can check that every rank holds exactly the right subdomains, in order.
    ranks_r: current low rank r of every subdomain (last entry of rank_hist).
    busy: measured busy time of every subdomain.
    """
    ids = list(range(bounds[rank], bounds[rank + 1]))
    busy = np.zeros(bounds[-1]) if busy is None else busy
    return dict(
        lrs=[f"lr{g}" for g in ids],
        grids=[SimpleNamespace(Nx=Nx, Nmu=Nmu, id=g) for g in ids],
        rngs=[f"rng{g}" for g in ids],
        timers=[SimpleNamespace(times={"PSI": float(busy[g])}) for g in ids],
        rank_hist=[[5, int(ranks_r[g])] for g in ids],
    )


### migrate

@pytest.mark.parametrize("old, new", [
    ([0, 2, 4, 6], [0, 2, 4, 6]),     # nothing changes
    ([0, 2, 4, 6], [0, 1, 4, 6]),     # rank 0 gives one to rank 1
    ([0, 1, 2, 8], [0, 3, 6, 8]),     # bad start, many move to the left
    ([0, 6, 7, 8], [0, 1, 2, 8]),     # rank 0 gives to rank 1 and further to rank 2
])
def test_migrate(old, new):
    """Every subdomain ends up on its new owner, in order, with all its data."""
    p = len(old) - 1
    ranks_r = np.arange(old[-1]) % 3 + 5
    migrated = run_ranks(p, lambda comm: migrate(
        comm, old, new, fake_domains(old, comm.Get_rank(), ranks_r)))
    for rank in range(p):
        ids = list(range(new[rank], new[rank + 1]))
        d = migrated[rank]
        assert [g.id for g in d["grids"]] == ids
        assert d["lrs"] == [f"lr{g}" for g in ids]
        assert d["rngs"] == [f"rng{g}" for g in ids]
        assert [h[-1] for h in d["rank_hist"]] == list(ranks_r[ids])


### Balancer.step

def run_balancer(p, start, ranks_r, steps, busy=None, **kw):
    """Balancer.step on all fake ranks, returns (balancer, domains) per rank."""
    def fn(comm):
        bal = Balancer(comm, start, **kw)
        d = fake_domains(start, comm.Get_rank(), ranks_r, busy=busy)
        for step in steps:
            d = bal.step(step, 0.1 * step, d)
        return bal, d
    return run_ranks(p, fn)


def test_model_weights_from_rank_hist():
    """
    "model" uses the current rank (last entry of rank_hist): the high-rank
    subdomains on the left get their own ranks.
    """
    ranks_r = np.array([30, 30, 5, 5, 5, 5, 5, 5])
    out = run_balancer(2, [0, 4, 8], ranks_r, steps=[0, 1], every=1, weights="model")
    for bal, d in out:
        assert bal.bounds == [0, 1, 8]
        assert bal.n_moved == 3
        assert bal.history == [(0.0, [0, 4, 8]), (0.1, [0, 1, 8])]
    assert [g.id for g in out[0][1]["grids"]] == [0]


def test_measured_weights_and_hysteresis():
    """A gain below min_gain does not migrate, a large one does."""
    # [0, 1, 4]: blocks 1 and 2.05, best split [0, 2, 4]: 1.05 and 2, only 2.4 % better
    busy = np.array([1.0, 0.05, 1.0, 1.0])
    out = run_balancer(2, [0, 1, 4], np.full(4, 5), steps=[1], busy=busy, every=1,
                       weights="measured", min_gain=0.05)
    assert all(bal.bounds == [0, 1, 4] and bal.n_moved == 0 for bal, _ in out)
    out = run_balancer(2, [0, 1, 4], np.full(4, 5), steps=[1], busy=busy, every=1,
                       weights="measured", min_gain=0.01)
    assert all(bal.bounds == [0, 2, 4] and bal.n_moved == 1 for bal, _ in out)

    busy = np.array([1.0, 1.0, 1.0, 3.0])
    out = run_balancer(2, [0, 2, 4], np.full(4, 5), steps=[1], busy=busy, every=1,
                       weights="measured", min_gain=0.05)
    assert all(bal.bounds == [0, 3, 4] and bal.n_moved == 1 for bal, _ in out)


def test_only_every_k_steps():
    ranks_r = np.array([30, 30, 5, 5, 5, 5, 5, 5])
    out = run_balancer(2, [0, 4, 8], ranks_r, steps=[0, 1, 2, 3], every=4,
                       weights="model")
    assert all(bal.events == [] and bal.bounds == [0, 4, 8] for bal, _ in out)


def test_summary_and_from_local():
    def fn(comm):
        return Balancer.from_local(comm, n_local=comm.Get_rank() + 1, every=3)
    for bal in run_ranks(3, fn):
        assert bal.bounds == [0, 1, 3, 6]
        s = bal.summary()
        assert s["every"] == 3 and s["n_changes"] == 0 and s["n_moved"] == 0
