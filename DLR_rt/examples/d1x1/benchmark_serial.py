"""
Serial benchmarks of the 1x1d DLR solver on one core.

    taskset -c 2 python -m DLR_rt.examples.d1x1.benchmark_serial nx
    taskset -c 2 python -m DLR_rt.examples.d1x1.benchmark_serial nmu
    taskset -c 2 python -m DLR_rt.examples.d1x1.benchmark_serial tol
    taskset -c 2 python -m DLR_rt.examples.d1x1.benchmark_serial domains
    python -m DLR_rt.examples.d1x1.benchmark_serial memory
    python -m DLR_rt.examples.d1x1.benchmark_serial plot nx      (only plot)

Results: data/benchmarks/serial_<experiment>.json, plots/serial_<experiment>.png
"""
import os

# BLAS and LAPACK to 1 thread, has to be set before numpy is imported.
# Always overwrite, the shell sets OMP_NUM_THREADS=22 globally
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"

import argparse
import json
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from DLR_rt.src.dd_1x1d import integrate_serial, setup_domains
from DLR_rt.src.sysinfo import print_system_info, system_info


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "benchmarks"
PLOT_DIR = ROOT / "plots" / "benchmark_serial"
BYTES = 8 # float64

#[128, 256, 512, 1024, 2048, 4096]

EXPERIMENTS = {
    "nx": [dict(Nx=Nx, Nmu=128) for Nx in [128, 256, 512, 1024, 2048, 4096]],
    "nmu": [dict(Nx=512, Nmu=Nmu) for Nmu in [32, 64, 128, 256, 512, 1024]],
    "tol": [dict(Nx=512, Nmu=256, tol=tol)
            for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]],
    "domains": [dict(Nx=2048, Nmu=128, n_domains=n) for n in [1, 2, 4, 8, 16, 32]],
    "memory": [dict(Nx=Nx, Nmu=256, repeat=1, trace_memory=True)
               for Nx in [256, 512, 1024, 2048, 4096]],
}

def measure(Nx, Nmu, n_domains=1, tol=1e-5, t_f=0.2, cfl=0.25, r0=5, seed=0,
            repeat=5, trace_memory=False) -> dict:
    """One configuration: warm-up, `repeat` timed runs, optional memory run."""
    def once():
        grids, lr0s, rngs = setup_domains(Nx, Nmu, r0, n_domains, seed)
        dt = cfl * grids[0].dx
        return integrate_serial(lr0s, grids, rngs, t_f, dt, tol, tol, progress=False)

    once()  # warm-up, not measured
    runs = [once() for _ in range(repeat)]
    steps = len(runs[0].time) - 1
    per_step = np.array([r.runtime / steps for r in runs])
    best = runs[int(per_step.argmin())]
    ranks = np.array(best.rank_hist)  # [domain][step]

    # Time per step and phase, summed over the domains
    phases = {}
    for tim in best.timings:
        for k, s in tim.items():
            phases[k] = phases.get(k, 0.0) + s / steps

    # Storage at the largest rank of each domain: U (Nx r), S (r r), V (Nmu r)
    r_max = ranks.max(axis=1)
    mem_lr = sum((g.Nx * r + r * r + Nmu * r) * BYTES
                 for g, r in zip(best.grids, r_max))

    rec = dict(
        Nx=Nx, Nmu=Nmu, n_domains=n_domains, tol=tol, t_f=t_f, steps=steps,
        repeat=repeat,
        t_step_min=float(per_step.min()),
        t_step_median=float(np.median(per_step)),
        t_step_all=per_step.tolist(),
        phases=phases,
        rank_mean=float(ranks.mean()),
        rank_max=int(ranks.max()),
        rank_final=ranks[:, -1].tolist(),
        mem_lowrank=int(mem_lr),
        mem_full=Nx * Nmu * BYTES,
    )

    if trace_memory:
        # Own run, tracemalloc makes the code much slower.
        # PhaseTimer resets the tracemalloc peak in every phase, so take the
        # extra memory per phase from res.mem_peak, not the global peak.
        tracemalloc.start()
        res = once()
        rec["mem_current_end"] = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        rec["mem_peak_phase"] = {k: max(m.get(k, 0) for m in res.mem_peak)
                                 for k in res.mem_peak[0]}
    return rec


def run(name: str):
    info = system_info()
    print_system_info(info)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"serial_{name}.json"
    records = []
    for cfg in EXPERIMENTS[name]:
        rec = measure(**cfg)
        records.append(rec)
        print(f"Nx={rec['Nx']:5d} Nmu={rec['Nmu']:5d} dom={rec['n_domains']:3d} "
              f"tol={rec['tol']:.0e}  {rec['t_step_min'] * 1e3:8.3f} ms/step "
              f"(median {rec['t_step_median'] * 1e3:.3f})  r_mean={rec['rank_mean']:.1f}")
        # Save after every configuration
        path.write_text(json.dumps(dict(system=info, records=records), indent=1))
    print(f"Saved {path}")
    plot(name)


def loglog_fit(ax, x, y, label, marker="o"):
    """Data and fitted slope, only on the upper half (small sizes = Python overhead)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    k = len(x) // 2
    slope, c = np.polyfit(np.log(x[k:]), np.log(y[k:]), 1)
    ax.loglog(x, y, marker + "-", label=label)
    ax.loglog(x, np.exp(c) * x**slope, "k--", lw=0.8, label=f"fit: slope {slope:.2f}")


def data_xticks(ax, x):
    """Ticks at the measured sizes with plain labels, no overlapping minor labels."""
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot(name: str):
    records = json.loads((DATA_DIR / f"serial_{name}.json").read_text())["records"]
    get = lambda key: [r[key] for r in records]  # noqa: E731
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if name in ("nx", "nmu"):
        x = get("Nx" if name == "nx" else "Nmu")
        xlabel = "$N_x$" if name == "nx" else r"$N_\mu$"
        loglog_fit(ax1, x, np.array(get("t_step_min")) * 1e3, "time per step")
        ax1.set(xlabel=xlabel, ylabel="time per step [ms]")
        data_xticks(ax1, x)
        # Phase breakdown as fractions, shows which part dominates
        phases = list(records[0]["phases"])
        bottom = np.zeros(len(x))
        for ph in phases:
            frac = np.array([r["phases"][ph] / sum(r["phases"].values()) for r in records])
            ax2.bar(range(len(x)), frac, bottom=bottom, label=ph)
            bottom += frac
        ax2.set(xticks=range(len(x)), xticklabels=x, xlabel=xlabel,
                ylabel="fraction of step time")
    elif name == "tol":
        loglog_fit(ax1, get("rank_mean"), np.array(get("t_step_min")) * 1e3,
                   "time per step")
        ax1.set(xlabel="mean rank $r$", ylabel="time per step [ms]")
        ax2.semilogx(get("tol"), get("rank_mean"), "o-", label="mean rank")
        ax2.semilogx(get("tol"), get("rank_max"), "s-", label="max rank")
        ax2.set(xlabel="tolerance", ylabel="rank")
        ax2.invert_xaxis()
    elif name == "domains":
        n, t = np.array(get("n_domains")), np.array(get("t_step_min"))
        ax1.semilogx(n, t[0] / t, "o-", base=2)
        ax1.axhline(1, color="k", lw=0.8)
        ax1.set(xlabel="number of subdomains (serial)", ylabel="speedup vs 1 domain")
        ax2.semilogx(n, get("rank_mean"), "o-", base=2, label="mean rank per domain")
        ax2.set(xlabel="number of subdomains (serial)", ylabel="rank")
    elif name == "memory":
        x = get("Nx")
        ax1.loglog(x, np.array(get("mem_full")) / 2**20, "s-", label="full $f$")
        ax1.loglog(x, np.array(get("mem_lowrank")) / 2**20, "o-", label="low-rank $U,S,V$")
        ax1.set(xlabel="$N_x$", ylabel="storage [MiB]")
        for ph in records[0]["mem_peak_phase"]:
            ax2.loglog(x, [r["mem_peak_phase"][ph] / 2**20 for r in records], "o-", label=ph)
        ax2.set(xlabel="$N_x$", ylabel="extra peak memory per phase [MiB]")
        data_xticks(ax1, x)
        data_xticks(ax2, x)

    for ax in (ax1, ax2):
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"serial {name}")
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / f"serial_{name}.png", dpi=200)
    print(f"Saved {PLOT_DIR / f'serial_{name}.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=[*EXPERIMENTS, "plot"])
    parser.add_argument("plot_experiment", nargs="?", choices=list(EXPERIMENTS))
    args = parser.parse_args()
    if args.experiment == "plot":
        plot(args.plot_experiment)
    else:
        run(args.experiment)