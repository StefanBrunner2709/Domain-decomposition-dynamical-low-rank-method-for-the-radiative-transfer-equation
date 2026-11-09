"""
Benchmarks for the MPI domain decomposition.


Note that p --> here is MPI with ranks p means the number of processes i.e the CPU cores
final_ranks --> are the numbers in the rank snapshows corresponding to the low-rank r of each subdomain



    mpirun -n 8 python -m DLR_rt.examples.d1x1.benchmark_mpi_scaling strong
    mpirun -n 8 python -m DLR_rt.examples.d1x1.benchmark_mpi_scaling weak
    mpirun -n 4 python -m DLR_rt.examples.d1x1.benchmark_mpi_scaling domains
    mpirun -n 8 python -m DLR_rt.examples.d1x1.benchmark_mpi_scaling asymmetric
    python -m DLR_rt.examples.d1x1.benchmark_mpi_scaling plot strong --inflow beam  (only plot)
    python -m DLR_rt.examples.d1x1.benchmark_mpi_scaling plot compare  (all strong runs,
                                                         imbalance and balance E(t))

strong:  fixed Nx and fixed decomposition (n_domains = max ranks), p = 1, 2, 4, ...
         Only the distribution of the domains changes, the result is identical,
         so the speedup is purely from the parallelisation.
weak:    Nx and n_domains grow with p, the work per rank stays the same.
domains: fixed number of ranks, more and more domains per rank.
asymmetric: fixed Nx and ranks, one domain per rank. Even split vs. more ranks at the
         physical boundaries vs. more ranks in the middle (smaller domains there).

Results are saved per inflow: data/benchmarks/strong_beam.json, plots/bench_strong_beam.png
"""

import os

# BLAS and LAPACK to 1 thread per MPI rank, has to be set before numpy is imported
# Always overwrite, the shell may set e.g. OMP_NUM_THREADS=22 globally
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


import argparse
import json
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np 
from mpi4py import MPI


from DLR_rt.src import plotting as P
from DLR_rt.src.dd_1x1d import (
    RunResult,
    beam_inflow,
    pulse_inflow,
    setup_domains,
    two_beam_inflow,
)
from DLR_rt.src.dd_mpi import domain_slice, gather_result, integrate_mpi
from DLR_rt.src.sysinfo import print_system_info, system_info

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "benchmarks"
INFLOWS = {"tanh": None, "beam": beam_inflow, "pulse": pulse_inflow,
           "two_beam": two_beam_inflow,
           "sharp_beam": partial(beam_inflow, mu_min=0.95, ramp=50.0)}
world = MPI.COMM_WORLD

def run(p: int , Nx: int, n_domains: int, args, sizes=None) -> RunResult | None:
    """ 
    Run on the first p ranks, take best of args.repeat runs

    Returns the Runresult on global rank 0, None on all other ranks
    """
    rank = world.Get_rank()
    sub = world.Split(0 if rank < p else MPI.UNDEFINED, rank)
    best = None
    if sub != MPI.COMM_NULL:
        dt = args.cfl / Nx  # dx = 1 / (Nx + 1), so dt <= cfl * dx
        for _ in range(args.repeat):
            grids, lr0s, rngs = setup_domains(Nx, args.Nmu, args.r, n_domains, args.seed,
                                              sizes=sizes)
            mine = domain_slice(n_domains, sub)
            sub.Barrier()
            local = integrate_mpi(lr0s[mine], grids[mine], rngs[mine], args.t_f, dt,
                                  sub, args.tol, args.tol, progress=False,
                                  inflow=INFLOWS[args.inflow])
            res = gather_result(sub, local)
            if res is not None and (best is None or res.runtime < best.runtime):
                best = res
        sub.Free()
    world.Barrier()
    return best

def rank_load(res: RunResult, p: int) -> np.ndarray:
    """ 
    Summed low-rank r of each MPI rank, shape [rank][time step]
    """
    hist, owner = np.array(res.rank_hist), np.array(res.owner)
    return np.array([hist[owner == i].sum(axis=0) for i in range(p)])

def rank_imbalance(res: RunResult, p: int) -> np.ndarray:
    """  
    Max / Mean over the MPI ranks of the summed low-rank r, for every time step
    
    As the cost grows slower than r, the real imbalance is smaller
    """
    load = rank_load(res, p)
    return load.max(axis=0) / load.mean(axis=0)

def rank_balance(res: RunResult, p: int) -> np.ndarray:
    """ 
    Min / Max over the MPI ranks of the summed low-rank r, 1 = perfectly balanced
    """
    load = rank_load(res, p)
    return load.min(axis=0) / load.max(axis=0)



def summary(res: RunResult, p: int, Nx: int, n_domains: int) -> dict:
    """ 
    The numbers of one run that are needed for scaling plots
    """
    per_rank = res.timings_per_rank()
    busy = [sum(s for k, s in tim.items() if k != "exchange") for tim in per_rank]
    steps = len(res.time) - 1
    every = max(1, steps // 200)  # enough points for a plot over time
    imbalance = max(busy) / np.mean(busy)  # 1 = all ranks equally busy
    balance = min(busy) / max(busy)  # E from the paper, 1 = perfectly balanced
    return dict(
        p=p, Nx=Nx, n_domains=n_domains, steps=steps,
        runtime=res.runtime, time_per_step=res.runtime / steps,
        busy=busy, exchange=[tim.get("exchange", 0.0) for tim in per_rank],
        imbalance=imbalance, balance=balance,
        final_ranks=[h[-1] for h in res.rank_hist],
        domains_per_rank=[res.owner.count(i) for i in range(p)],
        t_hist=res.time[::every],
        rank_imbalance=rank_imbalance(res, p)[::every].tolist(),
        rank_balance=rank_balance(res, p)[::every].tolist(),
    )

def plot_details(res: RunResult, name: str):
    """Solution at t_f, rank history, rank snapshots and timings per rank of one run."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    P.plot_solution(axes[0, 0], res)
    P.plot_rank_history(axes[0, 1], res)
    P.plot_rank_snapshots(axes[1, 0], res, times=np.linspace(0, res.time[-1], 11))
    P.plot_timings_per_rank(axes[1, 1], res)
    fig.suptitle(f"{name}: {res.n_domains} domains on {len(res.rank_timings)} ranks, "
                 f"Nx = {sum(g.Nx for g in res.grids)}")
    print(f"Saved {P.save_fig(fig, name)}")


def strong(args) -> list[dict]:
    n_domains = args.n_domains or max(args.ranks)
    records, ref = [], None
    for p in args.ranks:
        res = run(p, args.Nx, n_domains, args)
        if res is not None:
            if ref is None:
                ref = res  # first run (p = 1) is the serial reference
            rec = summary(res, p, args.Nx, n_domains)
            # Same decomposition, so f and the rank history have to be bit-identical
            rec["identical"] = bool(np.array_equal(res.f(), ref.f())
                                    and res.rank_hist == ref.rank_hist)
            records.append(rec)
            if p == max(args.ranks):
                plot_details(res, f"bench_strong_{args.inflow}_p{p}")
    return records

def weak(args) -> list[dict]:
    records = []
    for p in args.ranks:
        Nx, n_domains = args.Nx * p, (args.n_domains or 1) * p
        res = run(p, Nx, n_domains, args)
        if res is not None:
            records.append(summary(res, p, Nx, n_domains))
            if p == max(args.ranks):
                plot_details(res, f"bench_weak_{args.inflow}_p{p}")
    return records

def domains(args) -> list[dict]:
    p = world.Get_size()
    records = []
    for n_domains in args.domains or [p, 2 * p, 4 * p, 8 * p]:
        res = run(p, args.Nx, n_domains, args)
        if res is not None:
            records.append(summary(res, p, args.Nx, n_domains))
            plot_details(res, f"bench_domains_{args.inflow}_{n_domains}")
    return records

def layout_sizes(Nx: int, n_domains: int, refine: str) -> list[int]:
    """
    Mirror-symmetric subdomain sizes, with one domain per rank small domains = more ranks.

    refine = "boundary": small at both physical boundaries, e.g. 1 : 2 : 3 : 4 : 4 : ... : 1
    refine = "middle": exactly the other way around, e.g. 4 : 3 : 2 : 1 : 1 : ... : 4
    """
    weights = np.minimum(np.arange(1, n_domains + 1), np.arange(n_domains, 0, -1))
    if refine == "middle":
        weights = weights.max() + 1 - weights
    sizes = Nx * weights // weights.sum()
    # Remainder goes to the largest domains
    sizes[np.argsort(-weights, kind="stable")[:Nx - sizes.sum()]] += 1
    return sizes.tolist()

def asymmetric(args) -> list[dict]:
    """
    Even split vs. more ranks at the boundaries vs. more ranks in the middle.

    Every layout is also run on 1 rank, it has to be bit-identical.
    """
    p = max(args.ranks)
    n_domains = args.n_domains or p
    layouts = {"even": None,
               "boundary": layout_sizes(args.Nx, n_domains, "boundary"),
               "middle": layout_sizes(args.Nx, n_domains, "middle")}
    # Fewer grid points than r crashes in add_basis_functions, other ranks would hang
    assert min(layouts["boundary"]) >= args.r, "Every subdomain needs at least r grid points"

    records, f_even = [], None
    for layout, sizes in layouts.items():
        res = run(p, args.Nx, n_domains, args, sizes)
        ref = run(1, args.Nx, n_domains, args, sizes)
        if res is not None:
            f_even = res.f() if f_even is None else f_even
            diff = np.linalg.norm(res.f() - f_even) / np.linalg.norm(f_even)
            rec = summary(res, p, args.Nx, n_domains)
            rec.update(layout=layout, sizes=[g.Nx for g in res.grids],
                       rel_diff_even=float(diff),
                       identical=bool(np.array_equal(res.f(), ref.f())
                                      and res.rank_hist == ref.rank_hist))
            records.append(rec)
            print(f"{layout:>9}: sizes = {rec['sizes']}, f vs. even: {diff:.2e}")
            plot_details(res, f"bench_asymmetric_{layout}_{args.inflow}_p{p}")
    return records

def plot_asymmetric(records: list[dict], inflow: str):
    """Balance E(t) and busy time per MPI rank of all layouts of 'asymmetric'."""
    styles = ["-", "--", ":"]  # layouts stay apart without colour as well
    markers = ["o", "s", "^"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, rec in enumerate(records):
        axes[0].plot(rec["t_hist"], rec["rank_balance"], f"C{i}", ls=styles[i], lw=2,
                     label=rf"{rec['layout']} (busy $\mathcal{{E}}$ = {rec['balance']:.2f})")
        axes[1].plot(range(rec["p"]), rec["busy"], f"C{i}", ls=styles[i], lw=2,
                     marker=markers[i], ms=8,
                     label=f"{rec['layout']} ({rec['time_per_step'] * 1e3:.2f} ms/step)")
    ranks = range(records[0]["p"])
    axes[0].axhline(1.0, color="k", ls="--", lw=1)
    axes[0].set(xlabel="$t$", ylabel=r"$\mathcal{E}(t)$ = min / max of the rank load",
                ylim=(0, 1.05), title="Load balance over time (from the low-rank r)")
    axes[1].set(xlabel="MPI rank", ylabel="busy time [s]", ylim=(0, None), xticks=ranks,
                xticklabels=[f"P{i}" for i in ranks], title="Busy time per rank")
    for ax in axes:
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{records[0]['n_domains']} domains on {records[0]['p']} ranks, "
                 f"Nx = {records[0]['Nx']}, inflow {inflow}")
    print(f"Saved {P.save_fig(fig, f'bench_asymmetric_{inflow}')}")

def plot_summary(experiment: str, records: list[dict], inflow: str):
    if experiment == "asymmetric":
        plot_asymmetric(records, inflow)
        return

    p = np.array([rec["p"] for rec in records])
    t = np.array([rec["time_per_step"] for rec in records]) * 1e3
    imbalance = [rec["imbalance"] for rec in records]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if experiment == "strong":
        speedup = t[0] / t
        axes[0].plot(p, speedup, "o-", label="measured")
        axes[0].plot(p, p / p[0], "k--", lw=1, label="ideal")
        axes[0].set(xlabel="MPI ranks $p$", ylabel="speedup", xticks=p,
                    title=f"Strong scaling, Nx = {records[0]['Nx']}, "
                          f"{records[0]['n_domains']} domains")
        axes[1].plot(p, speedup / (p / p[0]) * 100, "o-")
        axes[1].set(xlabel="MPI ranks $p$", ylabel="parallel efficiency [%]",
                    xticks=p, ylim=(0, 110), title="Efficiency")
    elif experiment == "weak":
        axes[0].plot(p, t, "o-", label="measured")
        axes[0].axhline(t[0], color="k", ls="--", lw=1, label="ideal")
        axes[0].set(xlabel="MPI ranks $p$", ylabel="time per step [ms]", xticks=p,
                    ylim=(0, None), title=f"Weak scaling, Nx = {records[0]['Nx']} per rank")
        axes[1].plot(p, imbalance, "o-")
        axes[1].set(xlabel="MPI ranks $p$", ylabel="max busy / mean busy",
                    xticks=p, title="Load imbalance")
    else:
        n = np.array([rec["n_domains"] for rec in records])
        axes[0].plot(n, t, "o-")
        axes[0].set(xlabel="number of domains", ylabel="time per step [ms]", xticks=n,
                    ylim=(0, None), title=f"{p[0]} ranks, Nx = {records[0]['Nx']}")
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
    print(f"Saved {P.save_fig(fig, f'bench_{experiment}_{inflow}')}")


def plot_compare():
    """Imbalance and balance E(t) over time of the largest p of all strong_<inflow>.json."""
    print(f"\n{'inflow':>12} {'p':>3} {'busy imbalance':>15} {'busy E':>7} {'speedup':>8}"
          f"  final ranks")
    fig, ax = plt.subplots(figsize=(9, 5))
    fig_E, ax_E = plt.subplots(figsize=(9, 5))
    for i, inflow in enumerate(INFLOWS):
        path = DATA_DIR / f"strong_{inflow}.json"
        if path.exists():
            records = json.loads(path.read_text())["records"]
            rec = records[-1]
            ax.plot(rec["t_hist"], rec["rank_imbalance"], f"C{i}", lw=2, label=inflow)
            ax_E.plot(rec["t_hist"], rec["rank_balance"], f"C{i}", lw=2,
                      label=rf"{inflow} (busy $\mathcal{{E}}$ = {rec['balance']:.2f})")
            print(f"{inflow:>12} {rec['p']:>3} {rec['imbalance']:>15.2f} "
                  f"{rec['balance']:>7.2f} "
                  f"{records[0]['time_per_step'] / rec['time_per_step']:>8.2f}  "
                  f"{rec['final_ranks']}")
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set(xlabel="$t$", ylabel="max / mean of the rank load", ylim=(0.95, None),
           title="Load imbalance over time (from the low-rank r, not measured time)")
    ax.legend(fontsize=8)
    print(f"Saved {P.save_fig(fig, 'bench_compare_inflows')}")

    ax_E.axhline(1.0, color="k", ls="--", lw=1)
    ax_E.set(xlabel="$t$", ylabel=r"$\mathcal{E}(t)$ = min / max of the rank load",
             ylim=(0, 1.05),
             title="Load balance over time (from the low-rank r, not measured time)")
    ax_E.legend(fontsize=8, loc="lower right")
    print(f"Saved {P.save_fig(fig_E, 'bench_compare_balance')}")


def print_table(records: list[dict]):
    print(f"\n{'p':>3} {'Nx':>6} {'domains':>8} {'ms/step':>9} {'speedup':>8} "
          f"{'imbalance':>10} {'E':>6} {'identical':>10}  domains per rank, final ranks")
    t1 = records[0]["time_per_step"]
    for rec in records:
        print(f"{rec['p']:>3} {rec['Nx']:>6} {rec['n_domains']:>8} "
              f"{rec['time_per_step'] * 1e3:>9.2f} {t1 / rec['time_per_step']:>8.2f} "
              f"{rec['imbalance']:>10.2f} {rec['balance']:>6.2f} "
              f"{str(rec.get('identical', '-')):>10}  "
              f"{rec['domains_per_rank']}, {rec['final_ranks']}  {rec.get('layout', '')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment",
                        choices=["strong", "weak", "domains", "asymmetric", "plot"])
    parser.add_argument("plot_experiment", nargs="?", help="for 'plot': which one")
    parser.add_argument("--Nx", type=int, default=512, help="(per rank for weak)")
    parser.add_argument("--Nmu", type=int, default=64)
    parser.add_argument("--r", type=int, default=5, help="initial rank")
    parser.add_argument("--t_f", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.25)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inflow", choices=list(INFLOWS), default="tanh")
    parser.add_argument("--repeat", type=int, default=1, help="take the fastest run")
    parser.add_argument("--ranks", type=int, nargs="+", help="default 1, 2, 4, ... <= size")
    parser.add_argument("--n_domains", type=int,
                        help="strong: total (default max ranks), weak: per rank")
    parser.add_argument("--domains", type=int, nargs="+", help="for 'domains'")
    args = parser.parse_args()

    if args.experiment == "plot":
        if args.plot_experiment == "compare":
            plot_compare()
        else:
            path = DATA_DIR / f"{args.plot_experiment}_{args.inflow}.json"
            plot_summary(args.plot_experiment, json.loads(path.read_text())["records"],
                         args.inflow)
    else:
        size = world.Get_size()
        args.ranks = args.ranks or [p for p in (1, 2, 4, 8, 16, 32, 64) if p <= size]
        assert max(args.ranks) <= size, "Start mpirun with at least max(--ranks) ranks"

        # System info at the start (date, load), plus the CPUs each rank may run on,
        # this shows if the ranks were bound to cores (mpirun --bind-to core)
        rank_cpus = world.gather(sorted(os.sched_getaffinity(0)), root=0)
        if world.Get_rank() == 0:
            info = system_info()
            info["rank_cpus"] = rank_cpus
            print_system_info(info)

        experiment = {"strong": strong, "weak": weak, "domains": domains,
                      "asymmetric": asymmetric}[args.experiment]
        records = experiment(args)

        if world.Get_rank() == 0:
            print_table(records)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            path = DATA_DIR / f"{args.experiment}_{args.inflow}.json"
            path.write_text(json.dumps(dict(system=info, args=vars(args), records=records),
                                       indent=1))
            print(f"Saved {path}")
            plot_summary(args.experiment, records, args.inflow)