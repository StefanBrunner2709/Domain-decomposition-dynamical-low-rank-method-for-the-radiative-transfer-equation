"""
Serial benchmarks of the 1x1d DLR solver on one core.

    taskset -c 20 python -m DLR_rt.examples.d1x1.benchmark_serial nx
    taskset -c 20 python -m DLR_rt.examples.d1x1.benchmark_serial nmu
    taskset -c 20 python -m DLR_rt.examples.d1x1.benchmark_serial tol
    taskset -c 20 python -m DLR_rt.examples.d1x1.benchmark_serial domains
    taskset -c 20 python -m DLR_rt.examples.d1x1.benchmark_serial memory
    python -m DLR_rt.examples.d1x1.benchmark_serial plot nx      (only plot)
    python -m DLR_rt.examples.d1x1.benchmark_serial plot combined (nx time | memory)

Pin to a core outside the MPI range (mpirun binds rank i to core i), one run at a time.
Results: data/benchmarks/serial_<experiment>.json,
         plots/benchmark_serial/serial_<experiment>.png
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
MIB = 2**20

plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "legend.fontsize": 11})

EXPERIMENTS = {
    "nx": [dict(Nx=Nx, Nmu=128) for Nx in [128, 256, 512, 1024, 2048, 4096]],
    "nmu": [dict(Nx=512, Nmu=Nmu) for Nmu in [32, 64, 128, 256, 512, 1024]],
    "tol": [dict(Nx=512, Nmu=256, tol=tol)
            for tol in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]],
    "domains": [dict(Nx=2048, Nmu=128, n_domains=n) for n in [1, 2, 4, 8, 16, 32]],
    # Nx = Nmu: the full f grows like N^2, the low-rank storage like N
    "memory": [dict(Nx=N, Nmu=N, repeat=1, trace_memory=True)
               for N in [256, 512, 1024, 2048, 4096]],
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


def data_xticks(ax, x, labels=None):
    """Ticks at the measured sizes with plain labels, no overlapping minor labels."""
    ax.set_xticks(x)
    ax.set_xticklabels(labels or [str(v) for v in x])
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_times(ax, x, records):
    """Fastest of the repeats as point, in ms."""
    t = np.array([r["t_step_min"] for r in records]) * 1e3
    repeats = sorted({r["repeat"] for r in records})
    n_rep = "/".join(str(k) for k in repeats)  # normally one value, e.g. "5"
    ax.plot(x, t, "o", label=f"measured, fastest of {n_rep} repeats")
    return t


def linear_fit(ax, x, t, var: str, unit_b: str, scale_b: float = 1.0):
    """
    Fit t = a + b x and draw it: a is the fixed cost per step, b the cost per unit x.

    A power law on log-log axes would mix both parts into a meaningless slope.
    """
    if len(x) < 3:  # e.g. an interrupted run, a fit would be meaningless
        return None, None
    b, a = np.polyfit(np.asarray(x, float), t, 1)
    xx = np.linspace(0, 1.05 * max(x), 200)
    ax.plot(xx, a + b * xx, "k--", lw=1,
            label=f"fit $a + b\\,{var}$: $a$ = {a:.2f} ms, $b$ = {b * scale_b:.2f} {unit_b}")
    return a, b


def phase_bars(ax, records, ticklabels, xlabel):
    """Time per step and phase in ms, stacked (absolute, so the growth is visible)."""
    pos = np.arange(len(records))
    bottom = np.zeros(len(records))
    for ph in records[0]["phases"]:
        v = np.array([r["phases"][ph] for r in records]) * 1e3
        ax.bar(pos, v, bottom=bottom, label=ph)
        bottom += v
    ax.set(xticks=pos, xticklabels=ticklabels, xlabel=xlabel,
           ylabel="time per step (ms)", title="Time per phase")


def plot(name: str):
    records = json.loads((DATA_DIR / f"serial_{name}.json").read_text())["records"]
    get = lambda key: [r[key] for r in records]  # noqa: E731
    r0 = records[0]
    fixed = f"tol {r0['tol']:.0e}, $t_f$ = {r0['t_f']}, 1 core"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    if name in ("nx", "nmu"):
        if name == "nx":
            x, var, xlabel = get("Nx"), "N_x", "$N_x$"
            title = rf"Cost per step vs $N_x$ ($N_\mu$ = {r0['Nmu']}, {fixed})"
        else:
            x, var, xlabel = get("Nmu"), r"N_\mu", r"$N_\mu$"
            title = rf"Cost per step vs $N_\mu$ ($N_x$ = {r0['Nx']}, {fixed})"
        t = plot_times(ax1, x, records)
        linear_fit(ax1, x, t, var, r"µs/point", scale_b=1e3)
        ax1.set(xlabel=xlabel, ylabel="time per step (ms)", xlim=(0, None),
                ylim=(0, None), title="Time per step")
        phase_bars(ax2, records, x, xlabel)

    elif name == "tol":
        r, t = np.array(get("rank_mean")), plot_times(ax1, get("rank_mean"), records)
        # Fixed cost + rank-dependent part
        c2, c1, c0 = np.polyfit(r, t, 2)
        rr = np.linspace(0, 1.05 * r.max(), 200)
        ax1.plot(rr, c0 + c1 * rr + c2 * rr**2, "k--", lw=1,
                 label=f"fit $a + b r + c r^2$, $a$ = {c0:.2f} ms")
        ax1.set(xlabel=r"mean rank $\bar r$", ylabel="time per step (ms)",
                xlim=(0, None), ylim=(0, None), title="Time per step vs rank")
        ax2.semilogx(get("tol"), get("rank_mean"), "o-", label="mean rank")
        ax2.semilogx(get("tol"), get("rank_max"), "s-", label="max rank")
        ax2.set(xlabel="tolerance", ylabel="rank", title="Rank vs tolerance")
        ax2.invert_xaxis()
        title = (rf"Cost per step vs rank ($N_x$ = {r0['Nx']}, $N_\mu$ = {r0['Nmu']}, "
                 rf"$t_f$ = {r0['t_f']}, 1 core)")

    elif name == "domains":
        n = np.array(get("n_domains"))
        t = plot_times(ax1, n, records)
        # Cost per additional subdomain. Only as text: on the log2 axis a straight
        # line a + b n would look like an exponential curve.
        ax1.plot(n, t, "-", color="C0", lw=1, alpha=0.5)
        if len(n) >= 3:
            b = np.polyfit(n, t, 1)[0]
            ax1.text(0.03, 0.62, f"linear fit: {b:.2f} ms\nper additional subdomain",
                     transform=ax1.transAxes, fontsize=11,
                     bbox=dict(facecolor="white", edgecolor="0.7"))
        ax1.axhline(t[0], color="C2", ls=":", lw=1.5, label="1 domain")
        ax1.set_xscale("log", base=2)
        ticklabels = [f"{k}\n({r0['Nx'] // k})" for k in n]
        data_xticks(ax1, n, ticklabels)
        ax1.set(xlabel="subdomains $n$ (grid points each)", ylabel="time per step (ms)",
                ylim=(0, None), title="Time per step (serial)")
        ax2.plot(n, get("rank_mean"), "o-", label="mean rank per subdomain")
        ax2.set_xscale("log", base=2)
        data_xticks(ax2, n, ticklabels)
        ax2.set(xlabel="subdomains $n$ (grid points each)", ylabel="rank",
                title="Rank per subdomain")
        title = (rf"Serial domain decomposition ($N_x$ = {r0['Nx']}, "
                 rf"$N_\mu$ = {r0['Nmu']}, {fixed})")

    elif name == "memory":
        x = get("Nx")
        square = all(r["Nx"] == r["Nmu"] for r in records)
        xlabel = r"$N_x = N_\mu$" if square else rf"$N_x$ ($N_\mu$ = {r0['Nmu']})"
        lowrank = np.array(get("mem_lowrank")) / MIB
        peak = np.array([max(r["mem_peak_phase"].values()) for r in records]) / MIB
        ax1.loglog(x, np.array(get("mem_full")) / MIB, "s-", label="full $f$")
        ax1.loglog(x, lowrank + peak, "^-",
                   label="low-rank + largest extra peak of a phase")
        ax1.loglog(x, lowrank, "o-", label="low-rank storage $U, S, V$")
        ax1.set(xlabel=xlabel, ylabel="memory (MiB)", title="Storage and working memory")
        for ph in r0["mem_peak_phase"]:
            ax2.loglog(x, [r["mem_peak_phase"][ph] / MIB for r in records], "o-",
                       label=ph)
        ax2.set(xlabel=xlabel, ylabel="extra peak memory (MiB)",
                title="Extra peak memory per phase (tracemalloc)")
        data_xticks(ax1, x)
        data_xticks(ax2, x)
        title = f"Memory ({fixed})"

    for ax in (ax1, ax2):
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / f"serial_{name}.png", dpi=200)
    print(f"Saved {PLOT_DIR / f'serial_{name}.png'}")


def plot_combined():
    """Left: time per step vs N_x. Right: full f vs low-rank storage U, S, V."""
    load = lambda name: json.loads(  # noqa: E731
        (DATA_DIR / f"serial_{name}.json").read_text())["records"]
    rec_nx, rec_mem = load("nx"), load("memory")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    r0 = rec_nx[0]
    x = [r["Nx"] for r in rec_nx]
    t = plot_times(ax1, x, rec_nx)
    linear_fit(ax1, x, t, "N_x", r"µs/point", scale_b=1e3)
    ax1.set(xlabel="$N_x$", ylabel="time per step (ms)", xlim=(0, None),
            ylim=(0, None),
            title=rf"Time per step ($N_\mu$ = {r0['Nmu']}, tol = {r0['tol']:.0e}, "
                  rf"$t_f$ = {r0['t_f']})")

    m0 = rec_mem[0]
    x = [r["Nx"] for r in rec_mem]
    square = all(r["Nx"] == r["Nmu"] for r in rec_mem)
    xlabel = r"$N_x = N_\mu$" if square else rf"$N_x$ ($N_\mu$ = {m0['Nmu']})"
    ax2.loglog(x, np.array([r["mem_full"] for r in rec_mem]) / MIB, "s-",
               label="full $f$")
    ax2.loglog(x, np.array([r["mem_lowrank"] for r in rec_mem]) / MIB, "o-",
               label="low-rank storage $U, S, V$")
    data_xticks(ax2, x)
    ax2.set(xlabel=xlabel, ylabel="memory (MiB)",
            title=rf"Storage (tol = {m0['tol']:.0e}, $t_f$ = {m0['t_f']})")

    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / "serial_combined.png"
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=[*EXPERIMENTS, "plot"])
    parser.add_argument("plot_experiment", nargs="?",
                        choices=[*EXPERIMENTS, "combined"])
    args = parser.parse_args()
    if args.experiment == "plot" and args.plot_experiment == "combined":
        plot_combined()
    elif args.experiment == "plot":
        plot(args.plot_experiment)
    else:
        run(args.experiment)