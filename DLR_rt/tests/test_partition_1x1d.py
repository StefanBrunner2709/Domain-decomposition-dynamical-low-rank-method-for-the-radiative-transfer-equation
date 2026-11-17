"""Tests for DLR_rt/src/d1x1/partition.py."""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from DLR_rt.src.d1x1.partition import (
    _probe,
    balanced_bounds,
    balanced_bounds_brute,
    block_costs,
    block_imbalance,
    even_bounds,
    max_block,
    measured_weights,
    model_weights,
    owners,
    prefix_sum,
    slice_of,
)

W = np.array([1, 1, 1, 1, 50, 60, 1, 1, 1, 1], dtype=float)


def check_valid(bounds, n, p):
    """p contiguous blocks that cover 0..n, every rank gets at least one domain."""
    assert len(bounds) == p + 1
    assert bounds[0] == 0 and bounds[-1] == n
    assert all(a < b for a, b in zip(bounds, bounds[1:]))


def best_possible(w, p) -> float:
    """Smallest maximum block over all contiguous splits."""
    n = len(w)
    return min(max_block(w, [0, *cuts, n])
               for cuts in itertools.combinations(range(1, n), p - 1))


### Splits

# 10 on 4: the first 2 ranks get 3, 3 on 1: all on one rank
@pytest.mark.parametrize("n, p, sizes", [
    (4, 4, [1, 1, 1, 1]), (8, 4, [2, 2, 2, 2]), (10, 4, [3, 3, 2, 2]), (3, 1, [3]),
])
def test_even_bounds(n, p, sizes):
    bounds = even_bounds(n, p)
    check_valid(bounds, n, p)
    assert [slice_of(bounds, k).stop - slice_of(bounds, k).start
            for k in range(p)] == sizes


def test_even_bounds_too_many_ranks():
    with pytest.raises(AssertionError):
        even_bounds(2, 4)


@pytest.mark.parametrize("bounds, expected", [
    ([0, 2, 5], [0, 0, 1, 1, 1]), ([0, 1, 2, 3], [0, 1, 2]), ([0, 4], [0, 0, 0, 0]),
])
def test_owners(bounds, expected):
    np.testing.assert_array_equal(owners(bounds), expected)


### Cost of a split

def test_block_costs():
    assert block_costs(W, [0, 5, 6, 9, 10]).tolist() == [54.0, 60.0, 3.0, 1.0]
    assert max_block(W, [0, 5, 6, 9, 10]) == 60.0    # optimal split
    assert max_block(W, [0, 3, 6, 8, 10]) == 111.0   # even: 1 + 50 + 60 together
    assert max_block(np.arange(5.0), [0, 5]) == 10.0  # one rank: everything


def test_block_imbalance():
    assert block_imbalance(np.ones(8), even_bounds(8, 4)) == 1.0
    assert block_imbalance(W, [0, 3, 6, 8, 10]) == pytest.approx(111 / (118 / 4))


### Balanced splits

def test_prefix_sum():
    prefix = prefix_sum(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(prefix, [0.0, 1.0, 3.0, 6.0])
    assert prefix[3] - prefix[1] == 5.0


def test_probe_example():
    prefix = prefix_sum(W)
    assert _probe(prefix, 4, 60) == [0, 5, 6, 9, 10]
    assert _probe(prefix, 4, 59) is None


def test_balanced_example():
    """The optimal split is not unique, brute force returns the first it finds."""
    assert balanced_bounds(W, 4) == [0, 5, 6, 9, 10]
    assert balanced_bounds_brute(W, 4) == [0, 1, 5, 6, 10]
    assert max_block(W, balanced_bounds_brute(W, 4)) == 60.0


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("n, p", [(8, 3), (12, 4), (16, 5), (9, 9), (7, 1)])
def test_balanced_is_optimal(seed, n, p):
    """The bisection finds a split as good as the best of all splits."""
    w = np.random.default_rng(seed).lognormal(size=n)
    bounds = balanced_bounds(w, p)
    check_valid(bounds, n, p)
    best = best_possible(w, p)
    assert max_block(w, bounds) == pytest.approx(best, rel=1e-9)
    assert max_block(w, balanced_bounds_brute(w, p)) == best


def test_balanced_equal_weights_is_even():
    assert balanced_bounds(np.ones(12), 4) == even_bounds(12, 4)


def test_brute_too_many_splits():
    with pytest.raises(AssertionError):
        balanced_bounds_brute(np.ones(32), 8, max_splits=1000)


### Weights

def test_model_weights():
    grids = [SimpleNamespace(Nx=100, Nmu=28), SimpleNamespace(Nx=50, Nmu=28)]
    w = model_weights(grids, [[2], [2, 4]])
    assert w.tolist() == [128 * 4 + 8, np.mean([78 * 4 + 8, 78 * 16 + 64])]


def test_measured_weights():
    timings = [{"PSI": 1.0, "F_b": 0.5}, {"PSI": 2.0}]
    assert measured_weights(timings).tolist() == [1.5, 2.0]
