"""Tests for Grid_1x1d.split_n (DLR_rt/src/grid.py), moved from test_dd_1x1d."""
import numpy as np
import pytest

from DLR_rt.src.grid import Grid_1x1d


def test_split_n_2_equals_split():
    """split_n(2) gives exactly the same subgrids as split()."""
    grid = Grid_1x1d(256, 64, 5)
    for g_n, g in zip(grid.split_n(2), grid.split()):
        assert g_n.Nx == g.Nx
        assert np.array_equal(g_n.X, g.X)
        assert np.array_equal(g_n.MU, g.MU)
        assert g_n.dx == g.dx
        assert g_n.r == g.r


@pytest.mark.parametrize("Nx, n_split", [(256, 4), (100, 3), (10, 5)])
def test_split_n_covers_grid(Nx, n_split):
    """split_n covers the whole grid with disjoint subdomains."""
    grid = Grid_1x1d(Nx, 64, 5)
    grids = grid.split_n(n_split)
    assert len(grids) == n_split
    # Subdomains glued together give the original grid
    assert np.array_equal(np.concatenate([g.X for g in grids]), grid.X)
    # First Nx % n_split subdomains have one point more
    sizes = [g.Nx for g in grids]
    assert sizes == [Nx // n_split + (i < Nx % n_split) for i in range(n_split)]


def test_split_n_coeffs():
    """Every subdomain gets its own coefficient from _coeffs."""
    grids = Grid_1x1d(64, 16, 5).split_n(4, _coeffs=[1.0, 2.0, 3.0, 4.0])
    assert [g.coeff for g in grids] == [1.0, 2.0, 3.0, 4.0]


def test_split_n_too_many_domains():
    with pytest.raises(AssertionError):
        Grid_1x1d(6, 16, 5).split_n(4)


def test_split_n_sizes():
    """_sizes gives asymmetric subdomains that still cover the whole grid."""
    grid = Grid_1x1d(64, 16, 5)
    grids = grid.split_n(3, _sizes=[40, 4, 20])
    assert [g.Nx for g in grids] == [40, 4, 20]
    assert [len(g.X) for g in grids] == [40, 4, 20]
    assert np.array_equal(np.concatenate([g.X for g in grids]), grid.X)
    for g in grids:
        assert np.isclose(g.dx, grid.dx)


@pytest.mark.parametrize("sizes", [[32, 32], [30, 30, 3], [62, 1, 1], [64, 0, 0]])
def test_split_n_invalid_sizes(sizes):
    """Wrong length, wrong sum or a subdomain with fewer than 2 grid points."""
    with pytest.raises(AssertionError):
        Grid_1x1d(64, 16, 5).split_n(3, _sizes=sizes)
