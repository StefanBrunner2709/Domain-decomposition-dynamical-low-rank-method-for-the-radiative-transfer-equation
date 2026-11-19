"""
MPI scaling of the 1x1d domain decomposition.

Note: p is the number of MPI processes (CPU cores), the "ranks" in the records
(final_ranks, rank_hist) are the low ranks r of the subdomains.

    mpirun -n 16 python -m DLR_rt.benchmarks.d1x1.scaling strong --inflow beam
    mpirun -n 16 python -m DLR_rt.benchmarks.d1x1.scaling weak
    mpirun -n 4  python -m DLR_rt.benchmarks.d1x1.scaling domains
    mpirun -n 8  python -m DLR_rt.benchmarks.d1x1.scaling asymmetric
    python -m DLR_rt.benchmarks.d1x1.scaling plot strong --inflow beam --Nx 512
                                          (only plot, the same --inflow/--Nx/--suffix)
    python -m DLR_rt.benchmarks.d1x1.scaling plot compare --Nx 512
                                          (all strong runs with this Nx, imbalance
                                           and balance E(t))

strong:     fixed Nx and decomposition (n_domains = max p), p = 1, 2, 4, ...
            Only the distribution of the subdomains changes, the result is
            bit-identical, so the speedup is purely from the parallelisation.
weak:       Nx and n_domains grow with p, the work per process stays the same.
domains:    fixed p, more and more subdomains per process.
asymmetric: fixed Nx and p, one subdomain per process. Even split vs. smaller
            subdomains (more processes) at the physical boundaries vs. in the middle.

Results per experiment, inflow and Nx (Nx per process for weak):
    data/benchmarks/strong_beam_Nx512.json, plots/bench_strong_beam_Nx512.png
Clean timings: run alone, mpirun --bind-to core --map-by core ... --repeat 3
"""
from DLR_rt.benchmarks.d1x1 import common as C  # first, sets the BLAS threads

import argparse  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from DLR_rt.src.d1x1 import plotting as P  # noqa: E402
from DLR_rt.src.d1x1.inflow import INFLOWS  # noqa: E402


def case(args) -> dict:
    """Keyword arguments of C.run_mpi that are the same for all runs."""
    return dict(Nmu=args.Nmu, r=args.r, t_f=args.t_f, cfl=args.cfl, tol=args.tol,
                seed=args.seed, inflow=args.inflow, repeat=args.repeat)


def tag(args) -> str:
    return C.tag(args.inflow, args.Nx, args.suffix)


### Experiments, collective on all ranks, the records only on rank 0

def strong(args) -> list[dict]:
    n_domains = args.n_domains or max(args.ranks)
    # Best serial algorithm for the honest speedup: 1 subdomain on 1 process
    single = C.run_mpi(1, args.Nx, 1, **case(args))
    records, ref = [], None
    for p in args.ranks:
        res = C.run_mpi(p, args.Nx, n_domains, **case(args))
        if res is None:
            continue
        ref = ref or res  # the first run (p = 1) is the reference
        rec = C.summary(res, p, args.Nx, n_domains)
        t_single = single.runtime / (len(single.time) - 1)
        # Same decomposition, so f and the rank history have to be bit-identical
        rec.update(identical=C.identical(res, ref), t_single=t_single,
                   speedup_vs_single=t_single / rec["time_per_step"])
        records.append(rec)
        if p == max(args.ranks):
            C.plot_details(res, f"bench_strong_{tag(args)}_p{p}")
    return records


def weak(args) -> list[dict]:
    records = []
    for p in args.ranks:
        Nx, n_domains = args.Nx * p, (args.n_domains or 1) * p
        res = C.run_mpi(p, Nx, n_domains, **case(args))
        if res is not None:
            records.append(C.summary(res, p, Nx, n_domains))
            if p == max(args.ranks):
                C.plot_details(res, f"bench_weak_{tag(args)}_p{p}")
    return records


def domains(args) -> list[dict]:
    p = max(args.ranks)
    records = []
    for n_domains in args.domains or [p, 2 * p, 4 * p, 8 * p]:
        res = C.run_mpi(p, args.Nx, n_domains, **case(args))
        if res is not None:
            records.append(C.summary(res, p, args.Nx, n_domains))
            C.plot_details(res, f"bench_domains_{tag(args)}_{n_domains}")
    return records


def layout_sizes(Nx: int, n_domains: int, refine: str) -> list[int]:
    """
    Mirror-symmetric subdomain sizes, one subdomain per process.

    refine = "boundary": small at both physical boundaries, e.g. 1 : 2 : 3 : 4 : 4 : 3 : 2 : 1
    refine = "middle": the other way around, e.g. 4 : 3 : 2 : 1 : 1 : 2 : 3 : 4
    """
    weights = np.minimum(np.arange(1, n_domains + 1), np.arange(n_domains, 0, -1))
    if refine == "middle":
        weights = weights.max() + 1 - weights
    sizes = Nx * weights // weights.sum()
    # The remainder goes to the largest subdomains
    sizes[np.argsort(-weights, kind="stable")[:Nx - sizes.sum()]] += 1
    return sizes.tolist()


def asymmetric(args) -> list[dict]:
    """
    Even split vs. more processes at the boundaries vs. more in the middle.

    Every layout is also run on 1 process, it has to be bit-identical.
    """
    p = max(args.ranks)
    n_domains = args.n_domains or p
    layouts = {"even": None,
               "boundary": layout_sizes(args.Nx, n_domains, "boundary"),
               "middle": layout_sizes(args.Nx, n_domains, "middle")}
    # Fewer grid points than r crashes in add_basis, the other ranks would hang
    assert min(layouts["boundary"]) >= args.r, "Every subdomain needs >= r grid points"

    records, f_even = [], None
    for layout, sizes in layouts.items():
        res = C.run_mpi(p, args.Nx, n_domains, **case(args), sizes=sizes)
        # Only for the bit-identity check, one run is enough
        ref = C.run_mpi(1, args.Nx, n_domains, **{**case(args), "repeat": 1},
                        sizes=sizes)
        if res is None:
            continue
        f_even = res.f() if f_even is None else f_even
        diff = np.linalg.norm(res.f() - f_even) / np.linalg.norm(f_even)
        rec = C.summary(res, p, args.Nx, n_domains)
        rec.update(layout=layout, sizes=[g.Nx for g in res.grids],
                   rel_diff_even=float(diff), identical=C.identical(res, ref))
        records.append(rec)
        print(f"{layout:>9}: sizes = {rec['sizes']}, f vs. even: {diff:.2e}")
        C.plot_details(res, f"bench_asymmetric_{layout}_{tag(args)}_p{p}")
    return records


EXPERIMENTS = {"strong": strong, "weak": weak, "domains": domains,
               "asymmetric": asymmetric}


### Plots from the records, also for older result files

def plot_summary(experiment: str, records: list[dict], name: str):
    """Scaling plots of one experiment, name is the file tag, e.g. 'beam_Nx512'."""
    if experiment == "asymmetric":
        plot_asymmetric(records, name)
        return
    fig = summary_figure(experiment, records)
    print(f"Saved {P.save_fig(fig, f'bench_{experiment}_{name}')}")


def summary_figure(experiment: str, records: list[dict]):
    """The figure of plot_summary for strong, weak and domains, not saved."""
    p = np.array([rec["p"] for rec in records])
    t = np.array([rec["time_per_step"] for rec in records]) * 1e3
    imbalance = [rec["imbalance"] for rec in records]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if experiment == "strong":
        speedup = t[0] / t
        axes[0].plot(p, speedup, "o-", label="vs. same decomposition on 1 process")
        if records[0].get("speedup_vs_single") is not None:
            speedup_single = np.array([rec["speedup_vs_single"] for rec in records])
            axes[0].plot(p, speedup_single, "s--", label="vs. 1 domain on 1 process")
        axes[0].plot(p, p / p[0], "k--", lw=1, label="ideal")
        axes[0].set(xlabel="MPI processes $p$", ylabel="speedup", xticks=p,
                    title=f"Strong scaling, Nx = {records[0]['Nx']}, "
                          f"{records[0]['n_domains']} domains")
        # only vs. the same decomposition: vs. 1 domain it would be > 100 %
        axes[1].plot(p, speedup / (p / p[0]) * 100, "o-",
                     label="vs. same decomposition on 1 process")
        axes[1].set(xlabel="MPI processes $p$", ylabel="parallel efficiency (%)",
                    xticks=p, ylim=(0, 110), title="Efficiency")
    elif experiment == "weak":
        axes[0].plot(p, t, "o-", label="measured")
        axes[0].axhline(t[0], color="k", ls="--", lw=1, label="ideal")
        axes[0].set(xlabel="MPI processes $p$", ylabel="time per step (ms)", xticks=p,
                    ylim=(0, None),
                    title=f"Weak scaling, Nx = {records[0]['Nx']} per process")
        axes[1].plot(p, imbalance, "o-")
        axes[1].set(xlabel="MPI processes $p$", ylabel="max busy / mean busy",
                    xticks=p, title="Load imbalance")
    else:
        n = np.array([rec["n_domains"] for rec in records])
        axes[0].plot(n, t, "o-")
        axes[0].set(xlabel="number of domains", ylabel="time per step (ms)", xticks=n,
                    ylim=(0, None), title=f"{p[0]} processes, Nx = {records[0]['Nx']}")
        axes[1].plot(n, imbalance, "o-")
        axes[1].set(xlabel="number of domains", ylabel="max busy / mean busy",
                    xticks=n, title="Load imbalance")

    if experiment != "strong":
        # Fixed range from 1 (balanced), so small differences stay small
        axes[1].set_ylim(0.95, max(1.25, 1.05 * max(imbalance)))
        axes[1].axhline(1.0, color="k", ls="--", lw=1)
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    return fig


def plot_asymmetric(records: list[dict], name: str):
    """Balance E(t) and busy time per process of all layouts of 'asymmetric'."""
    styles = ["-", "--", ":"]  # the layouts stay apart without colour as well
    markers = ["o", "s", "^"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, rec in enumerate(records):
        axes[0].plot(rec["t_hist"], rec["rank_balance"], f"C{i}", ls=styles[i], lw=2,
                     label=rf"{rec['layout']} (busy $\mathcal{{E}}$ = "
                           rf"{rec['balance']:.2f})")
        axes[1].plot(range(rec["p"]), rec["busy"], f"C{i}", ls=styles[i], lw=2,
                     marker=markers[i], ms=8,
                     label=f"{rec['layout']} ({rec['time_per_step'] * 1e3:.2f} ms/step)")
    procs = range(records[0]["p"])
    axes[0].axhline(1.0, color="k", ls="--", lw=1)
    axes[0].set(xlabel="$t$",
                ylabel=r"$\mathcal{E}(t)$ = min / max of the summed $r$ per process",
                ylim=(0, 1.05), title="Load balance over time (from the low rank $r$)")
    axes[1].set(xlabel="process", ylabel="busy time (s)", ylim=(0, None), xticks=procs,
                xticklabels=[f"P{i}" for i in procs], title="Busy time per process")
    for ax in axes:
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{records[0]['n_domains']} domains on {records[0]['p']} processes, "
                 f"Nx = {records[0]['Nx']}, {name}")
    print(f"Saved {P.save_fig(fig, f'bench_asymmetric_{name}')}")


def plot_compare(Nx: int, suffix: str = ""):
    """Imbalance and balance E(t) at the largest p of all strong runs with this Nx."""
    print(f"\n{'inflow':>12} {'p':>3} {'busy imbalance':>15} {'busy E':>7} "
          f"{'speedup':>8}  final ranks")
    fig, ax = plt.subplots(figsize=(9, 5))
    fig_E, ax_E = plt.subplots(figsize=(9, 5))
    for i, inflow in enumerate(INFLOWS):
        path = C.DATA_DIR / f"strong_{C.tag(inflow, Nx, suffix)}.json"
        if not path.exists():
            continue
        records = C.load_json(path.stem)["records"]
        rec = records[-1]
        ax.plot(rec["t_hist"], rec["rank_imbalance"], f"C{i}", lw=2, label=inflow)
        ax_E.plot(rec["t_hist"], rec["rank_balance"], f"C{i}", lw=2,
                  label=rf"{inflow} (busy $\mathcal{{E}}$ = {rec['balance']:.2f})")
        print(f"{inflow:>12} {rec['p']:>3} {rec['imbalance']:>15.2f} "
              f"{rec['balance']:>7.2f} "
              f"{records[0]['time_per_step'] / rec['time_per_step']:>8.2f}  "
              f"{rec['final_ranks']}")
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set(xlabel="$t$", ylabel="max / mean of the summed $r$ per process",
           ylim=(0.95, None),
           title="Load imbalance over time (from the low rank $r$, not measured time)")
    ax.legend(fontsize=8)
    print(f"Saved {P.save_fig(fig, f'bench_compare_inflows_Nx{Nx}{suffix}')}")

    ax_E.axhline(1.0, color="k", ls="--", lw=1)
    ax_E.set(xlabel="$t$",
             ylabel=r"$\mathcal{E}(t)$ = min / max of the summed $r$ per process",
             ylim=(0, 1.05),
             title="Load balance over time (from the low rank $r$, not measured time)")
    ax_E.legend(fontsize=8, loc="lower right")
    print(f"Saved {P.save_fig(fig_E, f'bench_compare_balance_Nx{Nx}{suffix}')}")


def parser() -> argparse.ArgumentParser:
    ps = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ps.add_argument("experiment", choices=[*EXPERIMENTS, "plot"])
    ps.add_argument("plot_experiment", nargs="?",
                    choices=[*EXPERIMENTS, "compare"], help="for 'plot': which one")
    ps.add_argument("--Nx", type=int, default=512, help="(per process for weak)")
    ps.add_argument("--Nmu", type=int, default=64)
    ps.add_argument("--r", type=int, default=5, help="initial rank")
    ps.add_argument("--t_f", type=float, default=1.0)
    ps.add_argument("--cfl", type=float, default=0.25)
    ps.add_argument("--tol", type=float, default=1e-7)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--inflow", choices=list(INFLOWS), default="tanh")
    ps.add_argument("--repeat", type=int, default=1, help="take the fastest run")
    ps.add_argument("--ranks", type=int, nargs="+",
                    help="processes p, default 1, 2, 4, ... <= mpirun -n")
    ps.add_argument("--n_domains", type=int,
                    help="strong/asymmetric: total (default max p), weak: per process")
    ps.add_argument("--domains", type=int, nargs="+",
                    help="domains: numbers of subdomains, default p, 2p, 4p, 8p")
    ps.add_argument("--suffix", default="",
                    help="appended to all result file names (also for 'plot')")
    return ps


if __name__ == "__main__":
    args = parser().parse_args()

    if args.experiment == "plot":
        if args.plot_experiment == "compare":
            plot_compare(args.Nx, args.suffix)
        else:
            name = f"{args.plot_experiment}_{tag(args)}"
            plot_summary(args.plot_experiment, C.load_json(name)["records"], tag(args))
    else:
        from mpi4py import MPI

        world = MPI.COMM_WORLD
        size = world.Get_size()
        args.ranks = args.ranks or [p for p in (1, 2, 4, 8, 16, 32, 64) if p <= size]
        assert max(args.ranks) <= size, "Start mpirun with at least max(--ranks) ranks"

        info = C.start_mpi(world)
        records = EXPERIMENTS[args.experiment](args)
        if world.Get_rank() == 0:
            C.print_table(records)
            path = C.save_json(f"{args.experiment}_{tag(args)}", info, args=vars(args),
                               records=records)
            print(f"Saved {path}")
            plot_summary(args.experiment, records, tag(args))
