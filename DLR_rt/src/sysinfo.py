"""
System information for benchmark records.
This will be used for the methods section of the thesis

Usage at the start of a given benchmark:
 info = system_info()
 print_system_info(info)
 ... json.dumps(dict(system=info, records=recrods))
"""

import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

def _run(cmd: list[str]) -> str:
    """ 
    Output of a shell command, empty string if it fails
    """
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""

def _read(path: str) -> str:
    """ 
    Read a file, empty string if it fails
    """
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""

def _lscpu() -> dict:
    """ 
    Returns a dict of lscpu output, empty if it fails
    """
    info = {}
    for line in _run(["lscpu"]).splitlines():
        key, _, val = line.partition(":")
        info[key.strip()] = val.strip()
    return info

def system_info() -> dict:
    """Hardware, software, thread settings and code version of this run."""
    import numpy as np
    import scipy

    cpu = _lscpu()
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    meminfo = _read("/proc/meminfo").split()  # "MemTotal: <kB> kB ..."
    mem_kb = int(meminfo[1]) if len(meminfo) > 1 else 0
    info = dict(
        date=datetime.now().isoformat(timespec="seconds"),
        host=socket.gethostname(),
        command=" ".join(sys.argv),
        git_commit=_run(["git", "rev-parse", "--short", "HEAD"]),
        git_dirty=bool(_run(["git", "status", "--porcelain", "--untracked-files=no"])),
        cpu=cpu.get("Model name"),
        sockets=cpu.get("Socket(s)"),
        cores_per_socket=cpu.get("Core(s) per socket"),
        threads_per_core=cpu.get("Thread(s) per core"),
        numa_nodes=cpu.get("NUMA node(s)"),
        cpu_mhz_min_max=f"{cpu.get('CPU min MHz')} - {cpu.get('CPU max MHz')}",
        l3_cache=cpu.get("L3 cache"),
        governor=_read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        usable_cpus=len(os.sched_getaffinity(0)),
        load_avg_1min=os.getloadavg()[0],  # other users on the machine?
        ram_gb=round(mem_kb / 1024**2, 1),
        os=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        blas=f"{blas['name']} {blas['version']}",
        blas_config=blas.get("openblas configuration", ""),
        threads={k: os.environ.get(k) for k in
                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    )
    # Only if the benchmark already uses MPI, importing it here would start MPI
    if "mpi4py.MPI" in sys.modules:
        import mpi4py
        from mpi4py import MPI
        info.update(mpi4py=mpi4py.__version__,
                    mpi_library=MPI.Get_library_version().splitlines()[0],
                    mpi_size=MPI.COMM_WORLD.Get_size())
    return info


def print_system_info(info: dict):
    """Print info as a table, with warnings for conditions that spoil benchmarks."""
    width = max(map(len, info))
    print("=" * 60)
    for key, val in info.items():
        print(f"{key:<{width}} : {val}")
    print("=" * 60)
    if info["git_dirty"]:
        print("WARNING: uncommitted changes, commit before final benchmark runs")
    if info["load_avg_1min"] > 1.0:
        print(f"WARNING: machine is busy (load {info['load_avg_1min']:.1f})")

