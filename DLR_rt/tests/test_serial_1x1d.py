"""Tests for DLR_rt/benchmarks/d1x1/serial.py, small configurations only."""
import matplotlib

matplotlib.use("Agg")
import pytest  # noqa: E402

from DLR_rt.benchmarks.d1x1 import common as C  # noqa: E402
from DLR_rt.benchmarks.d1x1 import serial as S  # noqa: E402
from DLR_rt.src.d1x1 import plotting as P  # noqa: E402

SMALL = {
    "nx": [dict(Nx=Nx, Nmu=16, t_f=0.01, repeat=2) for Nx in (32, 64, 128)],
    "nmu": [dict(Nx=64, Nmu=Nmu, t_f=0.01, repeat=2) for Nmu in (8, 16, 32)],
    "tol": [dict(Nx=64, Nmu=16, tol=tol, t_f=0.01, repeat=2)
            for tol in (1e-3, 1e-5, 1e-7)],
    "domains": [dict(Nx=64, Nmu=16, n_domains=n, t_f=0.01, repeat=2)
                for n in (1, 2, 4)],
    "memory": [dict(Nx=N, Nmu=N, t_f=0.01, repeat=1, trace_memory=True)
               for N in (16, 32, 64)],
}


@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DATA_DIR", tmp_path)
    monkeypatch.setattr(P, "PLOTS_DIR", tmp_path)
    monkeypatch.setattr(S, "EXPERIMENTS", SMALL)
    return tmp_path


def test_measure_record():
    rec = S.measure(64, 16, n_domains=2, t_f=0.01, repeat=3)
    assert len(rec["t_step_all"]) == 3
    assert rec["t_step_min"] == min(rec["t_step_all"])
    assert set(rec["phases"]) == {"F_b", "add_basis", "PSI", "drop_basis"}
    assert rec["mem_full"] == 64 * 16 * 8 and len(rec["rank_final"]) == 2


def test_memory_record():
    rec = S.measure(32, 32, t_f=0.01, repeat=1, trace_memory=True)
    assert set(rec["mem_peak_phase"]) == {"F_b", "add_basis", "PSI", "drop_basis"}
    assert rec["mem_current_end"] > 0


@pytest.mark.parametrize("name", list(SMALL))
def test_run_and_plot(name, tmp_dirs):
    S.run(name)
    assert len(C.load_json(f"serial_{name}")["records"]) == 3
    assert (tmp_dirs / "benchmark_serial" / f"serial_{name}.png").exists()


def test_plot_combined(tmp_dirs):
    S.run("nx")
    S.run("memory")
    S.plot_combined()
    assert (tmp_dirs / "benchmark_serial" / "serial_combined.png").exists()
