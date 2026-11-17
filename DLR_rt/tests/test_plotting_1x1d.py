"""Smoke tests for DLR_rt/src/d1x1/plotting.py: every plot draws and saves."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from DLR_rt.src.d1x1 import plotting as P  # noqa: E402
from DLR_rt.src.d1x1.domains import integrate_serial, setup_domains  # noqa: E402
from DLR_rt.src.d1x1.inflow import beam  # noqa: E402

HISTORY = [(0.0, [0, 3, 5, 8]), (0.05, [0, 2, 5, 8]), (0.1, [0, 1, 4, 8])]


@pytest.fixture(scope="module")
def res():
    """Short serial run on 8 subdomains, dressed up as an MPI run on 3 processes."""
    grids, lrs, rngs = setup_domains(64, 16, 5, 8, seed=0)
    res = integrate_serial(lrs, grids, rngs, t_f=0.15, dt=1e-3, tol_sing_val=1e-5,
                           drop_tol=1e-5, inflow=beam, progress=False)
    res.owner = [0, 0, 0, 1, 1, 2, 2, 2]
    res.rank_timings = [{"exchange": 0.1}, {"exchange": 0.2}, {"exchange": 0.05}]
    res.balance = dict(every=10, weights="model", n_changes=2, n_moved=3,
                       history=HISTORY)
    return res


@pytest.fixture(autouse=True)
def tmp_plots(tmp_path, monkeypatch):
    """Save into a temporary directory instead of plots/."""
    monkeypatch.setattr(P, "PLOTS_DIR", tmp_path)
    return tmp_path


def test_helpers(res):
    assert len(P.interfaces(res)) == 7
    assert P.rank_blocks(res) == [3, 5]
    t, R = P.rank_matrix(res, [0.0, 0.15])
    assert R.shape == (2, 8) and t[0] == 0.0


def test_owner_matrix_and_bounds_steps():
    O = P.owner_matrix(HISTORY, [0.0, 0.07, 0.2], 8)
    assert O.tolist() == [[0, 0, 0, 1, 1, 2, 2, 2],
                          [0, 0, 1, 1, 1, 2, 2, 2],
                          [0, 1, 1, 1, 2, 2, 2, 2]]
    t, bounds = P.bounds_steps(HISTORY, 0.15)
    assert t == [0.0, 0.05, 0.1, 0.15]
    assert bounds.shape == (4, 4) and bounds[-1].tolist() == [0, 1, 4, 8]


@pytest.mark.parametrize("plot", [
    lambda ax, res: P.plot_solution(ax, res),
    lambda ax, res: P.plot_rank_history(ax, res),
    lambda ax, res: P.plot_rank_history(ax, res, max_legend=4),  # colour bar
    lambda ax, res: P.plot_rank_snapshots(ax, res, np.linspace(0, 0.15, 4)),
    lambda ax, res: P.plot_timings_per_rank(ax, res),
    lambda ax, res: P.plot_error_over_time(
        ax, np.arange(3), {"zero": np.zeros(3), "dd": np.array([1e-6, 1e-5, 1e-4])}),
])
def test_axis_plots(res, plot, tmp_plots):
    fig, ax = plt.subplots()
    plot(ax, res)
    path = P.save_fig(fig, "test")
    assert path == tmp_plots / "test.png" and path.stat().st_size > 0


def test_reassignment(res, tmp_plots):
    path = P.save_fig(P.plot_reassignment(res), "reassign")
    assert path.stat().st_size > 0
