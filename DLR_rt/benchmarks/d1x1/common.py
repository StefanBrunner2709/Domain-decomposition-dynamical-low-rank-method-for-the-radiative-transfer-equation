"""
Shared parts of the 1x1d benchmarks: runs, timing, records, system info, saving.

Import this module first in every benchmark, before numpy: it sets BLAS/LAPACK
to one thread (per MPI rank), which only works before numpy is loaded.

    from DLR_rt.benchmarks.d1x1 import common as C   # first import
    res = C.run_mpi(4, Nx=512, n_domains=8, Nmu=64, r=5, t_f=1.0, cfl=0.25,
                    tol=1e-7, inflow="beam", repeat=3)   # collective, rank 0 gets it
    rec = C.summary(res, 4, 512, 8)
    C.save_json("strong_beam_Nx512", info, records=[rec])

Clean timings: run alone on the machine, with
    mpirun -n 16 --bind-to core --map-by core python -m ... --repeat 3
"""
import os
import sys

# Always overwrite, the shell may set e.g. OMP_NUM_THREADS=22 globally
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"
if "numpy" in sys.modules:
    print("WARNING: numpy was imported before benchmarks.d1x1.common, "
          "BLAS may use more than one thread")

import json  # noqa: E402
import platform  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from DLR_rt.src.d1x1 import plotting as P  # noqa: E402
from DLR_rt.src.d1x1.domains import RunResult, integrate_serial, setup_domains  # noqa: E402
from DLR_rt.src.d1x1.inflow import INFLOWS  # noqa: E402
from DLR_rt.src.d1x1.partition import even_bounds, slice_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "benchmarks"


### Paths and saving

def tag(inflow: str, Nx: int, suffix: str = "", weights: str | None = None,
        n_domains: int | None = None) -> str:
    """
    Name part of all result files, e.g. 'beam_Nx512'.

    Balancing benchmarks add weights and the number of subdomains, e.g.
    'beam_Nx1024_measured_n32'. suffix keeps runs with other settings apart.
    """
    name = f"{inflow}_Nx{Nx}"
    if weights is not None:
        name += f"_{weights}" + (f"_n{n_domains}" if n_domains else "")
    return name + suffix


def save_json(name: str, system: dict, **content) -> Path:
    """data/benchmarks/<name>.json with the system info and content."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.json"
    path.write_text(json.dumps(dict(system=system, **content), indent=1))
    return path


def load_json(name: str) -> dict:
    """Content of data/benchmarks/<name>.json."""
    return json.loads((DATA_DIR / f"{name}.json").read_text())


### System info, for the methods section of the thesis

def _run(cmd: list[str]) -> str:
    """Output of a shell command, empty string if it fails."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _read(path: str) -> str:
    """Content of a file, empty string if it fails."""
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def system_info() -> dict:
    """Hardware, software, thread settings and code version of this run."""
    import scipy

    cpu = {}
    for line in _run(["lscpu"]).splitlines():
        key, _, val = line.partition(":")
        cpu[key.strip()] = val.strip()
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


def start_mpi(comm) -> dict | None:
    """
    System info at the start of an MPI benchmark, printed on rank 0, plus the CPUs
    every rank may run on: shows if the ranks were bound to cores. None on the others.
    """
    rank_cpus = comm.gather(sorted(os.sched_getaffinity(0)), root=0)
    if comm.Get_rank() != 0:
        return None
    info = system_info()
    info["rank_cpus"] = rank_cpus
    print_system_info(info)
    return info


### Runs

def dt_of(Nx: int, cfl: float) -> float:
    """dt = cfl dx with dx = 1 / (Nx + 1), the same for every decomposition."""
    return cfl / (Nx + 1)


def cpu_mhz() -> float:
    """Current clock of the first core this process may run on."""
    cpu = min(os.sched_getaffinity(0))
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
    try:
        return int(path.read_text()) / 1e3
    except OSError:
        return float("nan")  # not available on all systems, e.g. virtual machines


def wait_idle(comm):
    """Barrier that sleeps, so idle ranks do not use CPU time while others run."""
    req = comm.Ibarrier()
    while not req.Test():
        time.sleep(0.01)


def run_serial(Nx: int, n_domains: int, *, Nmu: int, r: int, t_f: float, cfl: float,
               tol: float, seed: int = 0, inflow: str = "tanh",
               sizes: list[int] | None = None, repeat: int = 1, warmup: bool = True,
               **kw) -> RunResult:
    """
    Serial domain decomposition (integrate_serial): warm-up, then repeat timed runs.

    Returns the fastest run, all runtimes in .extra["runtimes"].
    kw go to integrate_serial, e.g. use_neighbours or snap_every.
    """
    def once():
        grids, lrs, rngs = setup_domains(Nx, Nmu, r, n_domains, seed, sizes=sizes)
        return integrate_serial(lrs, grids, rngs, t_f, dt_of(Nx, cfl), tol, tol,
                                inflow=INFLOWS[inflow], progress=False, **kw)

    if warmup:
        once()
    runs = [once() for _ in range(repeat)]
    best = min(runs, key=lambda res: res.runtime)
    best.extra = dict(runtimes=[res.runtime for res in runs], mhz=[cpu_mhz()])
    return best


def run_mpi(p: int, Nx: int, n_domains: int, *, Nmu: int, r: int, t_f: float,
            cfl: float, tol: float, seed: int = 0, inflow: str = "tanh",
            sizes: list[int] | None = None, bounds: list[int] | None = None,
            balance: dict | None = None, repeat: int = 1, warmup: bool = True,
            comm=None, **kw) -> RunResult | None:
    """
    Run on the first p ranks of comm (default COMM_WORLD): a short warm-up, then
    repeat timed runs. Collective, every rank of comm has to call it.

    bounds: rank k gets the subdomains bounds[k]:bounds[k+1], default even split.
    balance: keyword arguments of Balancer (every, weights, min_gain), starts from
    bounds, a new Balancer for every run.
    kw go to integrate_mpi, e.g. use_neighbours or snap_every.

    Returns the fastest gathered RunResult on rank 0 of comm, with all runtimes and
    the clock of every rank in .extra, None on all other ranks.
    """
    from mpi4py import MPI

    from DLR_rt.src.d1x1.balancer import Balancer
    from DLR_rt.src.d1x1.mpi import gather_result, integrate_mpi

    comm = MPI.COMM_WORLD if comm is None else comm
    start = bounds if bounds is not None else even_bounds(n_domains, p)
    assert len(start) == p + 1 and start[-1] == n_domains, "Invalid bounds"
    dt = dt_of(Nx, cfl)
    rank = comm.Get_rank()
    sub = comm.Split(0 if rank < p else MPI.UNDEFINED, rank)
    best = None
    if sub != MPI.COMM_NULL:
        def once(t_end):
            grids, lrs, rngs = setup_domains(Nx, Nmu, r, n_domains, seed, sizes=sizes)
            mine = slice_of(start, sub.Get_rank())
            bal = Balancer(sub, start, **balance) if balance else None
            sub.Barrier()
            return integrate_mpi(lrs[mine], grids[mine], rngs[mine], t_end, dt, sub,
                                 tol, tol, inflow=INFLOWS[inflow], progress=False,
                                 balancer=bal, **kw)

        if warmup:  # a few steps: BLAS setup, MPI connections
            once(min(t_f, 20 * dt))
        runtimes = []
        for _ in range(repeat):
            local = once(t_f)
            mhz = sub.gather(cpu_mhz(), root=0)  # clock right after the run, per rank
            res = gather_result(sub, local)
            if res is not None:
                runtimes.append(res.runtime)
                if best is None or res.runtime < best.runtime:
                    best, best_mhz = res, mhz
        if best is not None:
            best.extra = dict(runtimes=runtimes, mhz=best_mhz)
        sub.Free()
    wait_idle(comm)
    return best


def identical(a: RunResult, b: RunResult) -> bool:
    """Same f and same rank history, bit by bit."""
    return bool(np.array_equal(a.f(), b.f()) and a.rank_hist == b.rank_hist)


### Records

def rank_load(res: RunResult, p: int) -> np.ndarray:
    """Summed low rank r of each process, shape [process][time step]."""
    hist, owner = np.array(res.rank_hist), np.array(res.owner or [0] * res.n_domains)
    return np.array([hist[owner == i].sum(axis=0) for i in range(p)])


def rank_imbalance(res: RunResult, p: int) -> np.ndarray:
    """
    Max / mean over the processes of the summed low rank r, for every time step.

    The cost grows faster than r, so this is not the measured imbalance.
    """
    load = rank_load(res, p)
    return load.max(axis=0) / load.mean(axis=0)


def rank_balance(res: RunResult, p: int) -> np.ndarray:
    """Min / max over the processes of the summed low rank r, 1 = balanced."""
    load = rank_load(res, p)
    return load.min(axis=0) / load.max(axis=0)


def lowrank_mib_per_rank(res: RunResult, p: int) -> list[float]:
    """
    Storage of U, S, V at the largest rank of each subdomain, summed per process.

    Computed instead of measured: the process peak (ru_maxrss) covers the whole
    lifetime and every rank also builds all initial conditions in setup_domains.
    """
    mem = np.zeros(p)
    for o, g, hist in zip(res.owner or [0] * res.n_domains, res.grids, res.rank_hist):
        r = max(hist)
        mem[o] += (g.Nx * r + r * r + g.Nmu * r) * 8 / 2**20
    return mem.tolist()


def summary(res: RunResult, p: int, Nx: int, n_domains: int) -> dict:
    """The numbers of one run for the JSON record and the plots."""
    per_rank = res.timings_per_rank()
    # Time in block_step per process. The subdomain timers would be wrong with
    # dynamic balancing (they move with the subdomain), fallback for serial results
    busy = res.rank_busy or [sum(s for k, s in tim.items()
                                 if k not in ("exchange", "rebalance"))
                             for tim in per_rank]
    steps = len(res.time) - 1
    every = max(1, steps // 200)  # enough points for a plot over time
    extra = getattr(res, "extra", {})
    rt = np.array(extra.get("runtimes") or [res.runtime])
    owner = res.owner or [0] * res.n_domains

    return dict(
        p=p, Nx=Nx, n_domains=n_domains, steps=steps,
        runtime=res.runtime, time_per_step=res.runtime / steps,
        busy=busy, exchange=[tim.get("exchange", 0.0) for tim in per_rank],
        rebalance=[tim.get("rebalance", 0.0) for tim in res.rank_timings],
        balancer=res.balance or None,
        imbalance=max(busy) / np.mean(busy),  # 1 = all processes equally busy
        balance=min(busy) / max(busy),  # E from the paper, 1 = perfectly balanced
        final_ranks=[h[-1] for h in res.rank_hist],
        domains_per_rank=[owner.count(i) for i in range(p)],
        t_hist=res.time[::every],
        rank_imbalance=rank_imbalance(res, p)[::every].tolist(),
        rank_balance=rank_balance(res, p)[::every].tolist(),
        runtimes=rt.tolist(),
        time_per_step_median=float(np.median(rt)) / steps,
        spread=float(rt.max() / rt.min() - 1.0),  # noise check, should be < ~5 %
        mhz=extra.get("mhz"),
        lowrank_mib=lowrank_mib_per_rank(res, p),
    )


def print_table(records: list[dict]):
    """One line per record: time per step, speedup vs. the first, imbalance."""
    print(f"\n{'p':>3} {'Nx':>6} {'domains':>8} {'ms/step':>9} {'spread':>7} "
          f"{'speedup':>8} {'imbalance':>10} {'E':>6} {'identical':>10}  "
          f"domains per rank, final ranks")
    t1 = records[0]["time_per_step"]
    for rec in records:
        print(f"{rec['p']:>3} {rec['Nx']:>6} {rec['n_domains']:>8} "
              f"{rec['time_per_step'] * 1e3:>9.2f} {rec['spread'] * 100:>6.1f}% "
              f"{t1 / rec['time_per_step']:>8.2f} "
              f"{rec['imbalance']:>10.2f} {rec['balance']:>6.2f} "
              f"{str(rec.get('identical', '-')):>10}  "
              f"{rec['domains_per_rank']}, {rec['final_ranks']}  "
              f"{rec.get('layout') or rec.get('split', '')}")


def plot_details(res: RunResult, name: str):
    """Solution at t_f, rank history, rank snapshots and timings per process."""
    print(f"Saved {P.save_fig(details_figure(res, name), name)}")


def details_figure(res: RunResult, name: str):
    """The figure of plot_details, not saved."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    P.plot_solution(axes[0, 0], res)
    P.plot_rank_history(axes[0, 1], res)
    P.plot_rank_snapshots(axes[1, 0], res, times=np.linspace(0, res.time[-1], 11))
    P.plot_timings_per_rank(axes[1, 1], res)
    fig.suptitle(f"{name}: {res.n_domains} domains on {len(res.rank_timings)} "
                 f"processes, Nx = {sum(g.Nx for g in res.grids)}")
    return fig
