"""
Tests for DLR_rt/benchmarks/d1x1/common.py.

run_mpi runs with p = 1 on COMM_WORLD, also without mpirun (singleton MPI).
"""
import json

import numpy as np
import pytest

from DLR_rt.benchmarks.d1x1 import common as C

CASE = dict(Nmu=32, r=5, t_f=0.05, cfl=0.25, tol=1e-6, inflow="beam")


@pytest.mark.parametrize("kw, name", [
    (dict(inflow="beam", Nx=512), "beam_Nx512"),
    (dict(inflow="tanh", Nx=2048, suffix="_n8_tf05"), "tanh_Nx2048_n8_tf05"),
    (dict(inflow="beam", Nx=1024, weights="measured", n_domains=32, suffix="_tf3"),
     "beam_Nx1024_measured_n32_tf3"),
    (dict(inflow="tanh", Nx=512, weights="model"), "tanh_Nx512_model"),
])
def test_tag_matches_existing_files(kw, name):
    assert C.tag(**kw) == name


def test_dt_of_is_cfl_dx():
    res = C.run_serial(128, 2, **CASE, warmup=False)
    assert np.isclose(C.dt_of(128, 0.25), 0.25 * res.grids[0].dx)
    assert res.time[1] == C.dt_of(128, 0.25)


def test_run_serial_fastest_of_repeat():
    res = C.run_serial(128, 2, **CASE, repeat=3)
    assert len(res.extra["runtimes"]) == 3
    assert res.runtime == min(res.extra["runtimes"])


@pytest.fixture(scope="module")
def mpi_res():
    pytest.importorskip("mpi4py")
    return C.run_mpi(1, 128, 4, **CASE, repeat=2,
                     balance=dict(every=5, weights="model"))


def test_run_mpi_p1_equals_serial(mpi_res):
    assert C.identical(mpi_res, C.run_serial(128, 4, **CASE, warmup=False))
    assert len(mpi_res.extra["runtimes"]) == 2
    assert mpi_res.owner == [0, 0, 0, 0]
    assert mpi_res.balance["every"] == 5


def test_run_mpi_invalid_bounds():
    pytest.importorskip("mpi4py")
    with pytest.raises(AssertionError):
        C.run_mpi(1, 128, 4, **CASE, bounds=[0, 3])


def test_summary(mpi_res):
    rec = C.summary(mpi_res, 1, 128, 4)
    json.dumps(rec)  # has to go into the JSON file
    assert rec["steps"] == len(mpi_res.time) - 1
    assert rec["domains_per_rank"] == [4]
    assert rec["imbalance"] == 1.0 and rec["balance"] == 1.0  # one process
    assert rec["rank_imbalance"][0] == 1.0
    assert rec["final_ranks"] == [h[-1] for h in mpi_res.rank_hist]
    assert rec["balancer"]["every"] == 5


def test_summary_serial_result():
    """Serial results have no owner and no rank_busy, the fallbacks are used."""
    res = C.run_serial(128, 3, **CASE, warmup=False)
    rec = C.summary(res, 1, 128, 3)
    assert rec["domains_per_rank"] == [3]
    assert rec["busy"][0] == pytest.approx(sum(sum(t.values()) for t in res.timings))


def test_rank_load():
    from types import SimpleNamespace
    res = SimpleNamespace(rank_hist=[[5, 6], [5, 9], [5, 5]], owner=[0, 0, 1],
                          n_domains=3)
    assert C.rank_load(res, 2).tolist() == [[10, 15], [5, 5]]
    np.testing.assert_allclose(C.rank_imbalance(res, 2), [10 / 7.5, 15 / 10])
    np.testing.assert_allclose(C.rank_balance(res, 2), [0.5, 5 / 15])


def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DATA_DIR", tmp_path)
    path = C.save_json("strong_x", {"host": "h"}, args={"Nx": 1}, records=[{"p": 1}])
    assert path == tmp_path / "strong_x.json"
    assert C.load_json("strong_x") == dict(system={"host": "h"}, args={"Nx": 1},
                                           records=[{"p": 1}])


def test_system_info_keys():
    info = C.system_info()
    for key in ("date", "git_commit", "numpy", "threads", "load_avg_1min"):
        assert key in info
    assert info["threads"]["OMP_NUM_THREADS"] == "1"
