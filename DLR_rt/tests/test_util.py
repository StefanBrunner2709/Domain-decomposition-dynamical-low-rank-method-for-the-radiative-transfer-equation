# util.computeD_upwind 1x1d
from DLR_rt.src.util import _upwind_matrices_1x1d, computeD_upwind_1x1d
from DLR_rt.src.grid import Grid_1x1d
import numpy as np

def test_upwind_matrices_dense():
    """
    Hand written test case
    """
    g = Grid_1x1d(4, 8, 2)
    DX_0, DX_1 = computeD_upwind_1x1d(g, "inflow")
    D0 = np.array([[1, 0, 0, 0], [-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]]) / g.dx
    D1 = np.array([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1], [0, 0, 0, -1]]) / g.dx
    np.testing.assert_allclose(DX_0.toarray(), D0)
    np.testing.assert_allclose(DX_1.toarray(), D1)

def test_upwind_periodic_corners():
    g = Grid_1x1d(4, 8, 2)
    DX_0, DX_1 = computeD_upwind_1x1d(g, "periodic")
    assert DX_0[0, -1] == -1 / g.dx
    assert DX_1[-1, 0] == 1 / g.dx

def test_upwind_cache_hit():
    _upwind_matrices_1x1d.cache_clear()
    a = computeD_upwind_1x1d(Grid_1x1d(32,8,2))
    b = computeD_upwind_1x1d(Grid_1x1d(32,8,2))
    c = computeD_upwind_1x1d(Grid_1x1d(64, 8, 2))
    assert a[0] is b[0]
    assert a[0] is not c[0]
    assert _upwind_matrices_1x1d.cache_info().hits == 1

