"""
Tests for DLR_rt/benchmarks/d1x1/scaling.py.

The experiments run in-process with p = 1 (singleton MPI, no mpirun), the full
runs with several processes are a manual mpirun of the benchmark.
"""
import matplotlib

matplotlib.use("Agg")
import pytest  # noqa: E402

pytest.importorskip("mpi4py")
from DLR_rt.benchmarks.d1x1 import common as C  # noqa: E402
from DLR_rt.benchmarks.d1x1 import scaling as S  # noqa: E402
from DLR_rt.src.d1x1 import plotting as P  # noqa: E402


@pytest.fixture(autouse=True)
def tmp_dirs(tmp_path, monkeypatch):
    """Results and plots into a temporary directory."""
    monkeypatch.setattr(C, "DATA_DIR", tmp_path)
    monkeypatch.setattr(P, "PLOTS_DIR", tmp_path)
    return tmp_path


def args_for(experiment, *extra):
    args = S.parser().parse_args([experiment, "--Nx", "64", "--Nmu", "16",
                                  "--t_f", "0.02", "--tol", "1e-6",
                                  "--inflow", "beam", *extra])
    args.ranks = [1]
    return args


@pytest.mark.parametrize("Nx, n", [(256, 4), (1000, 8), (101, 5)])
@pytest.mark.parametrize("refine", ["boundary", "middle"])
def test_layout_sizes(Nx, n, refine):
    sizes = S.layout_sizes(Nx, n, refine)
    assert sum(sizes) == Nx and len(sizes) == n
    assert sizes == sizes[::-1] or Nx % 2  # mirror-symmetric up to the remainder
    small = min(range(n), key=lambda i: sizes[i])
    assert (small in (0, n - 1)) == (refine == "boundary")


def test_tag():
    assert S.tag(args_for("strong")) == "beam_Nx64"
    assert S.tag(args_for("strong", "--suffix", "_x")) == "beam_Nx64_x"


@pytest.mark.parametrize("experiment, extra", [
    ("strong", []), ("weak", []), ("domains", ["--domains", "1", "2"]),
    ("asymmetric", []),
])
def test_experiment_and_plot(experiment, extra, tmp_dirs):
    args = args_for(experiment, *extra)
    records = S.EXPERIMENTS[experiment](args)
    assert records and all("time_per_step" in rec for rec in records)
    if experiment in ("strong", "asymmetric"):
        assert all(rec["identical"] for rec in records)
    S.plot_summary(experiment, records, S.tag(args))
    assert (tmp_dirs / f"bench_{experiment}_{S.tag(args)}.png").exists()


def test_plot_compare_reads_strong_files(tmp_dirs):
    args = args_for("strong")
    records = S.strong(args)
    C.save_json(f"strong_{S.tag(args)}", {}, records=records)
    S.plot_compare(64)
    assert (tmp_dirs / "bench_compare_inflows_Nx64.png").exists()
    assert (tmp_dirs / "bench_compare_balance_Nx64.png").exists()
