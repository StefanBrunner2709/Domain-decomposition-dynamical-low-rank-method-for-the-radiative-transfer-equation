import numpy as np
import pytest

from DLR_rt.src.grid import Grid_1x1d


def test_split_n_2_equals_split():
    """ 
    Comparison of both split_n and split functions
    """
    grid = Grid_1x1d(256, 64, 5)
    for g_n, g in zip(grid.split_n(2), grid.split()):
        assert np.allclose(g_n.X, g.X)
        assert np.allclose(g_n.MU, g.MU)
        assert g_n.r == g.r

@pytest.mark.parametrize("Nx, n_split", [(256, 4), (100, 3), (10, 5)])
def test_split_n_covers_grid(Nx, n_split):
    """ 
    Check that split_n covers the whole grid and that subdomains are disjoint
    """
    grid = Grid_1x1d(Nx, 64, 5)
    grids = grid.split_n(n_split)
    assert len(grids) == n_split
    # Subdomains glued together give the original grid
    assert np.array_equal(np.concatenate([g.X for g in grids]), grid.X)
    # first Nx % n_split subdomains have one point more
    sizes = [g.Nx for g in grids]
    assert sizes == [Nx // n_split + (i < Nx % n_split) for i in range(n_split)]

def test_split_n_coeffs():
    """ 
    Check that the coefficients of the subdomains are the same as the original grid
    """
    grids = Grid_1x1d(64, 16, 5).split_n(4, _coeffs=[1.0,2.0,3.0,4.0])
    assert [g.coeff for g in grids] == [1.0, 2.0, 3.0, 4.0]

def test_split_n_too_many_domains():
    with pytest.raises(AssertionError):
        Grid_1x1d(6,16,5).split_n(4)