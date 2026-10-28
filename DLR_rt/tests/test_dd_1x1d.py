import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from DLR_rt.src.dd_1x1d import full_f, integrate_serial, setup_domains
from DLR_rt.src.grid import Grid_1x1d
from DLR_rt.src.lr import computeF_b, computeF_b_edges, first_row, last_row
from DLR_rt.tests.make_reference import DT, NMU, NX, R, SEED, T_F, TOL
from DLR_rt.tests.make_reference import PATH as REFERENCE


### Grid_1x1d.split_n

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


### dd_1x1d: serial domain decomposition

def run_serial(n_domains, **kwargs):
    grids, lr0s, rngs = setup_domains(NX, NMU, R, n_domains, SEED)
    return integrate_serial(lr0s, grids, rngs, T_F, DT, TOL, TOL,
                            progress=False, **kwargs)


@pytest.fixture(scope="module")
def ref():
    """Reference results of the old scripts, see make_reference.py."""
    return np.load(REFERENCE)


@pytest.mark.parametrize("n_domains", [1, 2])
def test_serial_matches_reference(ref, n_domains):
    """Same result as the old dlr_inflow (1 domain) and dlr_dd (2 domains)."""
    res = run_serial(n_domains)
    # Exact in the same environment, allclose so another BLAS does not fail
    np.testing.assert_allclose(res.f(), ref[f"f_{n_domains}dom"], rtol=0, atol=1e-12)
    assert res.rank_hist == ref[f"rank_hist_{n_domains}dom"].tolist()


def test_serial_snapshots():
    res = run_serial(2, snap_every=50)
    assert res.snap_t[0] == 0.0
    assert len(res.snap_f) == len(res.snap_t)
    assert np.array_equal(res.snap_f[-1], res.f())


def test_serial_without_neighbours_differs():
    """Negative control: without the exchange the result has to change."""
    f = run_serial(2).f()
    f_ctrl = run_serial(2, use_neighbours=False).f()
    assert np.linalg.norm(f - f_ctrl) / np.linalg.norm(f) > 1e-2


def test_F_b_edges_equals_full():
    """computeF_b_edges gives the same F_b as computeF_b on the full matrices."""
    res = run_serial(2)
    (lr_l, lr_r), (g_l, g_r) = res.lrs, res.grids
    f_l, f_r = full_f(lr_l), full_f(lr_r)
    t = 0.5
    np.testing.assert_allclose(
        computeF_b_edges(t, lr_l, g_l, inflow_right=first_row(lr_r)),
        computeF_b(t, f_l, g_l, f_right=f_r), rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        computeF_b_edges(t, lr_r, g_r, inflow_left=last_row(lr_l)),
        computeF_b(t, f_r, g_r, f_left=f_l), rtol=0, atol=1e-14)


### dd_mpi

class FakeComm:
    """Only rank and size, enough for domain_slice."""

    def __init__(self, rank, size):
        self.rank, self.size = rank, size

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size


# 4 domains on 4 ranks: 1 each, 8 on 4: 2 each,
# 10 on 4: first 2 ranks get 3, last 2 ranks get 2, 3 on 1: all on one rank
@pytest.mark.parametrize("n_domains, size, sizes", [
    (4, 4, [1, 1, 1, 1]), (8, 4, [2, 2, 2, 2]), (10, 4, [3, 3, 2, 2]), (3, 1, [3]),
])
def test_domain_slice(n_domains, size, sizes):
    dd_mpi = pytest.importorskip("DLR_rt.src.dd_mpi")
    slices = [dd_mpi.domain_slice(n_domains, FakeComm(rank, size))
              for rank in range(size)]
    assert [s.stop - s.start for s in slices] == sizes
    # Contiguous blocks from left to right, covering all domains
    assert slices[0].start == 0 and slices[-1].stop == n_domains
    assert all(a.stop == b.start for a, b in zip(slices, slices[1:]))


def test_domain_slice_too_many_ranks():
    dd_mpi = pytest.importorskip("DLR_rt.src.dd_mpi")
    with pytest.raises(AssertionError):
        dd_mpi.domain_slice(2, FakeComm(0, 4))


# mpirun of the same environment as the running python, else from PATH
MPIRUN = (shutil.which("mpirun", path=str(Path(sys.executable).parent))
          or shutil.which("mpirun"))


@pytest.mark.skipif(MPIRUN is None, reason="mpirun not found")
@pytest.mark.parametrize("n_ranks, n_domains", [(2, 2), (2, 5)])
def test_mpi_matches_serial(n_ranks, n_domains):
    """Runs the check in dd_mpi.__main__, bit-identical to serial."""
    out = subprocess.run(
        [MPIRUN, "-n", str(n_ranks), sys.executable, "-m", "DLR_rt.src.dd_mpi",
         str(n_domains)],
        capture_output=True, text=True, timeout=300,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "All checks passed" in out.stdout
