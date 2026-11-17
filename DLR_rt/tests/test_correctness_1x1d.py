"""
Tests for DLR_rt/benchmarks/d1x1/correctness.py, small configurations only.

The full chain A - D and compare run with mpirun, see the module docstring.
"""
import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("mpi4py")
from DLR_rt.benchmarks.d1x1 import correctness as K  # noqa: E402
from DLR_rt.src.d1x1 import plotting as P  # noqa: E402

SMALL = dict(Nx=64, Nmu=16, r=5, seed=0, t_f=0.05, cfl=0.25, tol=1e-5)


@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PLOTS_DIR", tmp_path)
    return tmp_path


def test_A_original_solver_equals_restructured():
    """The original dlr_inflow and domains.integrate_serial (1 subdomain) agree."""
    f_orig, ranks_orig = K.original(SMALL)
    one = K.serial(SMALL, 1)
    out = K.compare(one.f(), f_orig, one.rank_hist, [ranks_orig])
    assert out["identical"] and out["same_ranks"] and out["max_abs"] == 0.0


def test_C_mpi_p1_equals_serial():
    res, ser = K.mpi(SMALL, 1, 4), K.serial(SMALL, 4)
    assert K.compare(res.f(), ser.f(), res.rank_hist, ser.rank_hist)["identical"]


def test_compare_and_rel_err():
    a, b = np.ones((3, 2)), np.full((3, 2), 2.0)
    out = K.compare(a, b)
    assert out == dict(identical=False, max_abs=1.0, rel_l2=0.5)
    assert K.compare(b, b, [[5, 6]], [[5, 6]])["same_ranks"]


def test_errors_over_time():
    """Relative error per snapshot, undefined (nan) where the reference is 0."""
    one = K.serial(SMALL, 1, snap_every=5)
    dd = K.serial(SMALL, 4, snap_every=5)
    err = K.errors(dd, one)
    assert np.isnan(err["rel_l2"][0])  # f(0) = 0
    assert np.all(err["rel_l2"][1:] > 0)
    assert np.all(K.errors(one, one)["max_abs"] == 0)


def test_plot_comparison(tmp_dirs):
    ser = K.serial(SMALL, 4, snap_every=5)
    mpi_res = K.mpi(SMALL, 1, 4, snap_every=5)
    single = K.serial(SMALL, 1, snap_every=5)
    e = K.errors(mpi_res, ser)
    K.plot_comparison(ser, mpi_res, single, e, K.errors(single, single),
                      K.errors(ser, single), "cmp")
    assert (tmp_dirs / "cmp.png").exists() and (tmp_dirs / "cmp.npz").exists()
