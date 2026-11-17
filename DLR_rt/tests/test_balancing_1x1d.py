"""
Tests for DLR_rt/benchmarks/d1x1/balancing.py.

The experiments run in-process with p = 1 (singleton MPI, no mpirun). The plots get
records copied to p = 2, 4. Migrations with several processes are tested by the
mpirun self-check of mpi.py (test_mpi_1x1d, dynamic).
"""
import matplotlib

matplotlib.use("Agg")
import pytest  # noqa: E402

pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from DLR_rt.benchmarks.d1x1 import balancing as B  # noqa: E402
from DLR_rt.benchmarks.d1x1 import common as C  # noqa: E402
from DLR_rt.src.d1x1 import plotting as P  # noqa: E402
from DLR_rt.src.d1x1.partition import even_bounds  # noqa: E402


@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DATA_DIR", tmp_path)
    monkeypatch.setattr(P, "PLOTS_DIR", tmp_path)
    return tmp_path


def args_for(experiment, *extra):
    args = B.parser().parse_args([experiment, "--Nx", "64", "--Nmu", "16",
                                  "--t_f", "0.02", "--tol", "1e-6", "--inflow", "beam",
                                  "--n_domains", "4", *extra])
    args.ranks = [1]
    return args


def test_tag():
    assert B.tag(args_for("dynamic")) == "beam_Nx64_measured_n4"
    assert B.tag(args_for("balanced", "--weights", "model", "--suffix", "_x")) \
        == "beam_Nx64_model_n4_x"


@pytest.mark.parametrize("weights", ["measured", "model"])
def test_balanced_reference(weights):
    records = B.balanced(args_for("balanced", "--weights", weights), MPI.COMM_WORLD)
    (rec,) = records
    assert rec["split"] == "even" and rec["identical"]
    assert len(rec["weights"]) == 4 and all(w > 0 for w in rec["weights"])


def test_dynamic_reference():
    (rec,) = B.dynamic(args_for("dynamic"), MPI.COMM_WORLD)
    assert rec["split"] == "even" and rec["identical"]


def fake_records(ref: dict, splits: list[str], balancer=None) -> list[dict]:
    """The 1-process record copied to p = 2, 4 for every split."""
    records = [ref]
    for p in (2, 4):
        for split in splits:
            bounds = even_bounds(4, p)
            rec = {**ref, "p": p, "split": split, "bounds": bounds,
                   "time_per_step": ref["time_per_step"] / p,
                   "predicted_imbalance": 1.0}
            if split.startswith("dynamic") and balancer:
                rec["balancer"] = {**balancer, "history": [(0.0, bounds)]}
            records.append(rec)
    return records


def test_plot_balanced(tmp_dirs):
    (ref,) = B.balanced(args_for("balanced"), MPI.COMM_WORLD)
    B.plot_balanced(fake_records(ref, ["even", "balanced"]), "x")
    assert (tmp_dirs / "bench_balanced_x.png").exists()


@pytest.mark.parametrize("with_balancer", [True, False])
def test_plot_dynamic(tmp_dirs, with_balancer):
    (ref,) = B.dynamic(args_for("dynamic"), MPI.COMM_WORLD)
    balancer = dict(every=5, n_moved=0) if with_balancer else None
    B.plot_dynamic(fake_records(ref, ["even", "dynamic 5"], balancer), "x")
    assert (tmp_dirs / "bench_dynamic_x.png").exists()
