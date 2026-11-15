""" 
Test for the MPI implementation of the 1x1d DD low-rank integrator.

Further also test for the load-balanced distribution of different subsdomains to MPI ranks 
"""
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

dd_mpi = pytest.importorskip("DLR_rt.src.dd_mpi")


### Helpers
def block_costs(w,bounds) -> list[float]:
    """ 
    Computes the summed weight of every block
    """
    return [float(np.sum(w[a:b])) for a,b in zip(bounds, bounds[1:])]

def best_possible(w, p) -> float:
    """Smallest maximum block cost over all contiguous splits (brute force)."""
    n = len(w)
    return min(max(block_costs(w, [0, *cuts, n]))
               for cuts in itertools.combinations(range(1, n), p - 1))


def check_valid(bounds, n, p):
    """p contiguous blocks that cover 0..n, every rank gets at least one domain."""
    assert len(bounds) == p + 1
    assert bounds[0] == 0 and bounds[-1] == n
    assert all(a < b for a, b in zip(bounds, bounds[1:]))

class FakeComm:
    """ 
    Only rank and size which is enough for the slice function
    """
    def __init__(self, rank, size):
        self.rank = rank
        self.size = size

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size


def test_prefix_sum():
    prefix = dd_mpi.prefix_sum(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(prefix, [0.0, 1.0, 3.0, 6.0])
    assert prefix[3] - prefix[1] == 5.0 #block 1..2 via prefix sum

def test_example():
    prefix = dd_mpi.prefix_sum(np.array([1, 1, 1, 1, 50, 60, 1, 1, 1, 1], dtype=float))
    assert dd_mpi._probe(prefix, 4, 60) == [0, 5, 6, 9, 10]
    assert dd_mpi._probe(prefix, 4, 59) is None