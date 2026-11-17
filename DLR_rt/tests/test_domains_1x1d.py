"""Tests for DLR_rt/src/d1x1/domains.py, the serial domain decomposition."""
import numpy as np
import pytest

from DLR_rt.src.d1x1.domains import (
    PhaseTimer,
    RunResult,
    block_step,
    integrate_serial,
    setup_domains,
)
from DLR_rt.src.d1x1.inflow import beam
from DLR_rt.tests.make_reference import DT, NMU, NX, R, SEED, T_F, TOL
from DLR_rt.tests.make_reference import PATH as REFERENCE


def run_serial(n_domains, **kwargs):
    grids, lrs, rngs = setup_domains(NX, NMU, R, n_domains, SEED)
    return integrate_serial(lrs, grids, rngs, T_F, DT, TOL, TOL,
                            progress=False, **kwargs)


@pytest.mark.parametrize("n_domains", [1, 2])
def test_matches_reference(n_domains):
    """Same result as the old dlr_inflow (1 domain) and dlr_dd (2 domains)."""
    ref = np.load(REFERENCE)
    res = run_serial(n_domains)
    # Exact in the same environment, allclose so another BLAS does not fail
    np.testing.assert_allclose(res.f(), ref[f"f_{n_domains}dom"], rtol=0, atol=1e-12)
    assert res.rank_hist == ref[f"rank_hist_{n_domains}dom"].tolist()


def test_setup_independent_of_n_domains():
    """The subdomains are glued together, initial f does not depend on the split."""
    _, lrs, _ = setup_domains(64, 16, 5, 1, seed=3)
    f1 = lrs[0].full()
    for n in (2, 4):
        grids, lrs, rngs = setup_domains(64, 16, 5, n, seed=3)
        assert len(grids) == len(lrs) == len(rngs) == n
        assert sum(g.Nx for g in grids) == 64
        assert all(lr.r == 5 for lr in lrs)
        assert np.all(np.abs(np.concatenate([lr.full() for lr in lrs])) < 1e-12)
    assert np.all(np.abs(f1) < 1e-12)  # random U, V but S = 0: f(0) = 0


def test_snapshots():
    res = run_serial(2, snap_every=50)
    assert res.snap_t[0] == 0.0
    assert len(res.snap_f) == len(res.snap_t)
    assert np.isclose(res.snap_t[-1], T_F)
    assert np.array_equal(res.snap_f[-1], res.f())


def test_without_neighbours_differs():
    """Negative control: without the exchange the result has to change."""
    f = run_serial(2).f()
    f_ctrl = run_serial(2, use_neighbours=False).f()
    assert np.linalg.norm(f - f_ctrl) / np.linalg.norm(f) > 1e-2


def test_inflow_changes_result_and_rank():
    """The beam enters only on the left: the left subdomain needs the higher rank."""
    res = run_serial(4, inflow=beam)
    assert not np.allclose(res.f(), run_serial(4).f())
    final = [h[-1] for h in res.rank_hist]
    assert final[0] > final[-1]


def test_block_step_outer_rows():
    """
    A block with outer_left/outer_right from the neighbours gives the same as the
    whole domain in one block, this is what the MPI version relies on.
    """
    grids, lrs, rngs = setup_domains(NX, NMU, R, 4, SEED)
    _, lrs_b, rngs_b = setup_domains(NX, NMU, R, 4, SEED)
    timers = [PhaseTimer() for _ in range(4)]
    t, dt = 0.0, DT
    for _ in range(20):
        # reference: all 4 in one block
        block_step(lrs, grids, rngs, timers, t, dt, TOL, TOL, inflow=beam)
        # split into blocks [0, 1] and [2, 3], edge rows from the state before
        mid_left, mid_right = lrs_b[1].last_row(), lrs_b[2].first_row()
        left, right = lrs_b[:2], lrs_b[2:]
        block_step(left, grids[:2], rngs_b[:2], timers[:2], t, dt, TOL, TOL,
                   inflow=beam, outer_right=mid_right)
        block_step(right, grids[2:], rngs_b[2:], timers[2:], t, dt, TOL, TOL,
                   inflow=beam, outer_left=mid_left)
        lrs_b[:] = left + right
        t += dt
    for a, b in zip(lrs, lrs_b):
        assert a.r == b.r
        assert np.array_equal(a.full(), b.full())


def test_timings_per_rank_sums_domains():
    """3 domains, the first two on rank 0: their phases add up, exchange added."""
    res = RunResult(lrs=[], grids=[None] * 3, time=[0.0], rank_hist=[],
                    timings=[{"PSI": 1.0}, {"PSI": 2.0}, {"PSI": 4.0}],
                    mem_peak=[], owner=[0, 0, 1],
                    rank_timings=[{"exchange": 0.5}, {"exchange": 0.1}])
    per_rank = res.timings_per_rank()
    assert per_rank[0] == {"exchange": 0.5, "PSI": 3.0}
    assert per_rank[1] == {"exchange": 0.1, "PSI": 4.0}


def test_timer_phases():
    res = run_serial(2)
    for tim in res.timings:
        assert set(tim) == {"F_b", "add_basis", "PSI", "drop_basis"}
        assert all(s > 0 for s in tim.values())


def test_save_roundtrip(tmp_path):
    res = run_serial(2, snap_every=50)
    res.save(tmp_path / "run.npz")
    data = np.load(tmp_path / "run.npz")
    np.testing.assert_array_equal(data["f"], res.f())
    assert data["rank_hist"].tolist() == res.rank_hist
    assert len(data["X"]) == NX
