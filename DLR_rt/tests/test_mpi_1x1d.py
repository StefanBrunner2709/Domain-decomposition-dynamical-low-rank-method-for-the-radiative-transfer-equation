"""
Tests for DLR_rt/src/d1x1/mpi.py.

The MPI runs start mpirun with the self-check in mpi.__main__, which compares the
gathered result with integrate_serial bit by bit.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

mpi = pytest.importorskip("DLR_rt.src.d1x1.mpi")
from mpi4py import MPI  # noqa: E402

# mpirun of the same environment as the running python, else from PATH
MPIRUN = (shutil.which("mpirun", path=str(Path(sys.executable).parent))
          or shutil.which("mpirun"))


class FakeComm:
    def __init__(self, rank, size):
        self.rank, self.size = rank, size

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size


def test_neighbours():
    assert mpi.neighbours(FakeComm(0, 1)) == (MPI.PROC_NULL, MPI.PROC_NULL)
    assert mpi.neighbours(FakeComm(0, 3)) == (MPI.PROC_NULL, 1)
    assert mpi.neighbours(FakeComm(1, 3)) == (0, 2)
    assert mpi.neighbours(FakeComm(2, 3)) == (1, MPI.PROC_NULL)


@pytest.mark.skipif(MPIRUN is None, reason="mpirun not found")
@pytest.mark.parametrize("n_ranks, args", [
    (2, ["2"]),               # one subdomain per rank
    (2, ["5"]),               # uneven blocks 3 + 2
    (3, ["8", "brute"]),      # optimal split from measured weights
    (3, ["8", "balanced"]),   # the same with the bisection
    (3, ["12", "dynamic"]),   # bad start, the balancer migrates subdomains
])
def test_mpi_matches_serial(n_ranks, args):
    out = subprocess.run(
        [MPIRUN, "-n", str(n_ranks), sys.executable, "-m", "DLR_rt.src.d1x1.mpi",
         *args],
        capture_output=True, text=True, timeout=300,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "All checks passed" in out.stdout
