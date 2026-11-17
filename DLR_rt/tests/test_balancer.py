""" 
Test for the dynamic load balancer

Collective calls of balancer with allgather, stuff like this are tested without mpirun
"""

import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

balancer = pytest.importorskip("DLR_rt.src.balancer")
dd_mpi = pytest.importorskip("DLR_rt.src.dd_mpi")
from mpi4py import MPI

from DLR_rt.src.balancer import Balancer, max_block, migrate, owners

### Helper class for the MPI communicator
class ThreadComm:
    """ 
    Fake communicator for p ranks, every rank runs in its own thread
    """
    def __init__(self, rank, size, shared):
        self.rank = rank
        self.size = size
        self.shared = rank, size, shared

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def _exchange(self,obj):
        """ 
        Every rank puts obj, in every rank gets the list of all p
        """
        sh = self.shared
        sh["buf"][self.rank] = obj
        sh["barrier"].wait()
        out = list(sh["buf"])
        sh["barrier"].wait()
        return out

    def allgather(self, obj):
        return self._exchange(obj)

    def alltoall(self, send):
        """ 
        Rank j recieves send[j] of every rank
        """
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

def fake_domains(bounds, rank, ranks_r, Nx=64, Nmu=64):
    """
    Local subdomains of rank for the split bounds, without a real simulation.

    Every subdomain g is labelled with its global index, so after a migration we
    can check that every rank holds exactly the right subdomains, in order.
    ranks_r: low-rank r of every subdomain, used by the "model" weights.
    """
    ids = list(range(bounds[rank], bounds[rank + 1]))
    return dict(
        lrs=[f"lr{g}" for g in ids],
        grids=[SimpleNamespace(Nx=Nx, Nmu=Nmu, r=ranks_r[g], id=g) for g in ids],
        rngs=[f"rng{g}" for g in ids],
        timers=[SimpleNamespace(times={"PSI": 0.0}) for _ in ids],
        rank_hist=[[ranks_r[g]] for g in ids],
    )

### owners

@pytest.mark.parametrize("bounds, expected", [
    ([0, 2, 5], [0, 0, 1, 1, 1]),
    ([0, 1, 2, 3], [0, 1, 2]),
    ([0, 4], [0, 0, 0, 0]),
])
def test_owners(bounds, expected):
    np.testing.assert_array_equal(owners(bounds, bounds[-1]), expected)


## max_block

def test_max_block():
    w = np.array([1, 1, 1, 1, 50, 60, 1, 1, 1, 1], dtype=float)
    assert max_block(w, [0, 5, 6, 9, 10]) == 60.0   # optimal split
    assert max_block(w, [0, 3, 6, 8, 10]) == 111.0  # even split: 1 + 50 + 60 together


def test_max_block_one_rank():
    w = np.arange(5, dtype=float)
    assert max_block(w, [0, 5]) == w.sum()


@pytest.mark.parametrize("old, new", [
    ([0, 2, 4, 6], [0, 2, 4, 6]),     # nothing changes
    ([0, 2, 4, 6], [0, 1, 4, 6]),     # rank 0 gives one to rank 1
    ([0, 1, 2, 8], [0, 3, 6, 8]),     # bad start, many move to the left
    ([0, 6, 7, 8], [0, 1, 2, 8]),     # rank 0 gives to rank 1 and further to rank 2
])
def test_migrate(old, new):
    """Check that migrate moves the right subdomains to the right ranks."""
    p = len(old) - 1
    ranks_r = np.arange(old[-1]) % 3  # low-rank r of every subdomain
    domains = [fake_domains(old, rank, ranks_r) for rank in range(p)]
    migrated = migrate(domains, old, new)
    for rank in range(p):
        ids = list(range(new[rank], new[rank + 1]))
        assert [g.id for g in migrated[rank]["grids"]] == ids
        assert [g.r for g in migrated[rank]["grids"]] == list(ranks_r[ids]) 
