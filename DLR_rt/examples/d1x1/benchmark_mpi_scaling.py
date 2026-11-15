"""
Benchmarks for the MPI domain decomposition.


Note that p --> here is MPI with ranks p means the number of processes i.e the CPU cores
final_ranks --> are the numbers in the rank snapshows corresponding to the low-rank r of each subdomain



    mpirun -n 8 python -m DLR_rt.examples.d1x1.benchmark_mpi_scaling strong
    mpirun -n 8 python -m DLR_rt.examples.d1x1.benchmark_mpi_scaling weak
    mpirun -n 4 python -m DLR_rt.examples.d1x1.benchmark_mpi_scaling domains
    mpirun -n 8 python -m DLR_rt.examples.d1x1.benchmark_mpi_scaling asymmetric
    mpirun -n 8 python -m DLR_rt.examples.d1x1.benchmark_mpi_scaling balanced --n_domains 32
    python -m DLR_rt.examples.d1x1.benchmark_mpi_scaling plot strong --inflow beam --Nx 512
                                                         (only plot, same --inflow/--Nx)
    python -m DLR_rt.examples.d1x1.benchmark_mpi_scaling plot compare --Nx 512
                                                         (all strong runs with this Nx,
                                                         imbalance and balance E(t))

Clean timings: run alone on the machine, with
    mpirun -n 16 --bind-to core --map-by core ... --repeat 3

strong:  fixed Nx and fixed decomposition (n_domains = max ranks), p = 1, 2, 4, ...
         Only the distribution of the domains changes, the result is identical,
         so the speedup is purely from the parallelisation.
weak:    Nx and n_domains grow with p, the work per rank stays the same.
domains: fixed number of ranks, more and more domains per rank.
asymmetric: fixed Nx and ranks, one domain per rank. Even split vs. more ranks at the
         physical boundaries vs. more ranks in the middle (smaller domains there).
balanced: fixed Nx and n_domains > p. Even split vs. balanced split of the domains onto
         the ranks (weights from the 1-rank run, dd_mpi.balanced_bounds), for every p.
dynamic: like balanced, but the split is recomputed during the run every --every steps
         and the subdomains migrate (balancer.Balancer), vs. the static even split.
         mpirun -n 8 ... dynamic --n_domains 32 --every 10 50 200

Results are saved per inflow and Nx (Nx per rank for weak):
    data/benchmarks/strong_beam_Nx512.json, plots/bench_strong_beam_Nx512.png
"""

import os

# BLAS and LAPACK to 1 thread per MPI rank, has to be set before numpy is imported
# Always overwrite, the shell may set e.g. OMP_NUM_THREADS=22 globally
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


import argparse
import json
import math
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np 
import time

from mpi4py import MPI


from DLR_rt.src import plotting as P
from DLR_rt.src.balancer import Balancer
from DLR_rt.src.dd_1x1d import (
    RunResult,
    beam_inflow,
    pulse_inflow,
    setup_domains,
    two_beam_inflow,
)
from DLR_rt.src.dd_mpi import (
    balanced_bounds,
    balanced_bounds_brute,
    block_imbalance,
    domain_slice,
    even_bounds,
    gather_result,
    integrate_mpi,
    measured_weights,
    model_weights,
)
from DLR_rt.src.sysinfo import print_system_info, system_info

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "benchmarks"
INFLOWS = {"tanh": None, "beam": beam_inflow, "pulse": pulse_inflow,
           "two_beam": two_beam_inflow,
           "sharp_beam": partial(beam_inflow, mu_min=0.95, ramp=50.0)}
world = MPI.COMM_WORLD

def wait_idle(comm):
    """ 
    Barrier sleeps, idle ranks do not consume CPU time
    """
    req = comm.Ibarrier()
    while not req.Test():
        time.sleep(0.01)  # 10 ms, not too small to avoid busy waiting

def cpu_mhz() -> float:
    """ 
    Current clock of the core this rank is bound to
    """
    cpu = min(os.sched_getaffinity(0))
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
    try:
        return int(path.read_text()) / 1e3
    except OSError:
        return float("nan")  # not available on all systems, e.g. some virtual machines


def tag(args) -> str:
    """Name part of all result files, e.g. 'beam_Nx512' (Nx per rank for weak)."""
    name = f"{args.inflow}_Nx{args.Nx}"
    if {"balanced", "dynamic"} & {args.experiment, args.plot_experiment}:
        name += f"_{args.weights}" + (f"_n{args.n_domains}" if args.n_domains else "")
    return name


def run(p: int , Nx: int, n_domains: int, args, sizes=None,
        bounds: list[int] | None = None, balance: dict | None = None) -> RunResult | None:
    """
    Run on the first p ranks: a short warm-up, then args.repeat timed runs.

    bounds: rank k gets the subdomains bounds[k]:bounds[k+1] (same on all ranks),
    default is the even split of domain_slice.
    balance: dynamic load balancing, keyword arguments of Balancer (every, weights),
    starts from bounds. A new Balancer for every run.

    Returns the fastest RunResult on global rank 0, with all runtimes and the clock
    of every rank in .extra, None on all other ranks.
    """
    rank = world.Get_rank()
    sub = world.Split(0 if rank < p else MPI.UNDEFINED, rank)
    best = None
    if sub != MPI.COMM_NULL:
        dt = args.cfl / Nx  # dx = 1 / (Nx + 1), so dt <= cfl * dx
        if bounds is not None:
            assert len(bounds) == p + 1 and bounds[-1] == n_domains, "Invalid bounds"

        def once(t_f):
            grids, lr0s, rngs = setup_domains(Nx, args.Nmu, args.r, n_domains, args.seed, sizes=sizes)
            start = bounds if bounds is not None else even_bounds(n_domains, p)
            mine = slice(start[sub.Get_rank()], start[sub.Get_rank() + 1])
            bal = Balancer(sub, start, **balance) if balance else None
            sub.Barrier()
            return integrate_mpi(lr0s[mine], grids[mine], rngs[mine], t_f, dt, sub,
                                 args.tol, args.tol, progress=False,
                                 inflow=INFLOWS[args.inflow], balancer=bal)

        # Warm-up, a few steps are enough: caching matrices, BLAS setup, MPI connections
        once(min(args.t_f, 20 * dt))
        runtimes = []
        for _ in range(args.repeat):
            local = once(args.t_f)
            mhz = sub.gather(cpu_mhz(), root=0)  # clock right after the run, per rank
            res = gather_result(sub, local)
            if res is not None:
                runtimes.append(res.runtime)
                if best is None or res.runtime < best.runtime:
                    best, best_mhz = res, mhz
        if best is not None:
            best.extra = dict(runtimes=runtimes, mhz=best_mhz)
        sub.Free()
    wait_idle(world)  # idle ranks sleep instead of spinning
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


def lowrank_mib_per_rank(res: RunResult, p: int) -> list[float]:
    """
    Storage of U, S, V at the largest rank of each domain, summed per MPI rank [MiB]

    Computed instead of measured: the process peak (ru_maxrss) covers the whole
    lifetime and every rank also builds all initial conditions in setup_domains.
    """
    mem = np.zeros(p)
    for o, g, hist in zip(res.owner, res.grids, res.rank_hist):
        r = max(hist)
        mem[o] += (g.Nx * r + r * r + g.Nmu * r) * 8 / 2**20
    return mem.tolist()



def summary(res: RunResult, p: int, Nx: int, n_domains: int) -> dict:
    """ 
    The numbers of one run that are needed for scaling plots
    """
    per_rank = res.timings_per_rank()
    # Time in block_step per rank. The subdomain timers would be wrong with dynamic
    # balancing (they move with the subdomain), fallback for results without it
    busy = res.rank_busy or [sum(s for k, s in tim.items()
                                 if k not in ("exchange", "rebalance"))
                             for tim in per_rank]
    steps = len(res.time) - 1
    every = max(1, steps // 200)  # enough points for a plot over time
    imbalance = max(busy) / np.mean(busy)  # 1 = all ranks equally busy
    balance = min(busy) / max(busy)  # E from the paper, 1 = perfectly balanced
    extra = getattr(res, "extra", {})
    rt = np.array(extra.get("runtimes") or [res.runtime])

    return dict(
        p=p, Nx=Nx, n_domains=n_domains, steps=steps,
        runtime=res.runtime, time_per_step=res.runtime / steps,
        busy=busy, exchange=[tim.get("exchange", 0.0) for tim in per_rank],
        rebalance=[tim.get("rebalance", 0.0) for tim in res.rank_timings],
        balancer=res.balance or None,
        imbalance=imbalance, balance=balance,
        final_ranks=[h[-1] for h in res.rank_hist],
        domains_per_rank=[res.owner.count(i) for i in range(p)],
        t_hist=res.time[::every],
        rank_imbalance=rank_imbalance(res, p)[::every].tolist(),
        rank_balance=rank_balance(res, p)[::every].tolist(),
        runtimes=rt.tolist(),
        time_per_step_median=float(np.median(rt)) / steps,
        spread=float(rt.max() / rt.min() - 1.0),  # noise check, should be < ~5 %
        mhz=extra.get("mhz"),
        lowrank_mib=lowrank_mib_per_rank(res, p),
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
    # Best serial algorithm: 1 domain on 1 rank. run is collective, all ranks call it.
    single = run(1, args.Nx, 1, args)
    t_single = single.runtime / (len(single.time) - 1) if single is not None else None

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
            # Honest end-to-end speedup against the best serial run (1 domain)
            rec["t_single"] = t_single
            rec["speedup_vs_single"] = t_single / rec["time_per_step"]
            records.append(rec)
            if p == max(args.ranks):
                plot_details(res, f"bench_strong_{tag(args)}_p{p}")
    return records

def weak(args) -> list[dict]:
    records = []
    for p in args.ranks:
        Nx, n_domains = args.Nx * p, (args.n_domains or 1) * p
        res = run(p, Nx, n_domains, args)
        if res is not None:
            records.append(summary(res, p, Nx, n_domains))
            if p == max(args.ranks):
                plot_details(res, f"bench_weak_{tag(args)}_p{p}")
    return records

def domains(args) -> list[dict]:
    p = world.Get_size()
    records = []
    for n_domains in args.domains or [p, 2 * p, 4 * p, 8 * p]:
        res = run(p, args.Nx, n_domains, args)
        if res is not None:
            records.append(summary(res, p, args.Nx, n_domains))
            plot_details(res, f"bench_domains_{tag(args)}_{n_domains}")
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

    # The 1-rank reference is only for the bit-identity check, one run is enough
    args_ref = argparse.Namespace(**{**vars(args), "repeat": 1})
    records, f_even = [], None
    for layout, sizes in layouts.items():
        res = run(p, args.Nx, n_domains, args, sizes)
        ref = run(1, args.Nx, n_domains, args_ref, sizes)
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
            plot_details(res, f"bench_asymmetric_{layout}_{tag(args)}_p{p}")
    return records

def balanced(args) -> list[dict]:
    """
    Even vs. balanced split of the same decomposition (n_domains > p), for every p.

    1. p = 1: reference for the speedup and the bit-identity check. Its busy time per
       subdomain gives the weights, the cost of a subdomain does not depend on which
       rank computes it (--weights model: cost model with the ranks of this run).
    2. For every p > 1: even split (domain_slice) and balanced split (balanced_bounds,
       smallest maximum block sum of the weights). The result is bit-identical,
       only the distribution of the subdomains differs.
    """
    n_domains = args.n_domains or 2 * max(args.ranks)
    ref = run(1, args.Nx, n_domains, args)
    weights = None
    if ref is not None:
        weights = (measured_weights(ref) if args.weights == "measured"
                   else model_weights(ref.grids, ref.rank_hist))
    weights = world.bcast(weights, root=0)  # same weights, so same bounds on every rank

    records = []
    if ref is not None:
        rec = summary(ref, 1, args.Nx, n_domains)
        rec.update(split="even", bounds=[0, n_domains], predicted_imbalance=1.0,
                   identical=True, weights=list(map(float, weights)))
        records.append(rec)

    for p in [p for p in args.ranks if p > 1]:
        splits = {"even": even_bounds(n_domains, p),
                  "balanced": balanced_bounds(weights, p)}
        for split, bounds in splits.items():
            if split == "balanced" and bounds == splits["even"]:
                # Nothing to balance, the run would be the same (same on all ranks)
                if world.Get_rank() == 0:
                    records.append({**records[-1], "split": "balanced"})
                continue
            res = run(p, args.Nx, n_domains, args, bounds=bounds)
            if res is None:
                continue
            rec = summary(res, p, args.Nx, n_domains)
            rec.update(split=split, bounds=bounds,
                       predicted_imbalance=block_imbalance(weights, bounds),
                       identical=bool(np.array_equal(res.f(), ref.f())
                                      and res.rank_hist == ref.rank_hist))
            if split == "balanced" and math.comb(n_domains - 1, p - 1) <= 200_000:
                # Bisection has to reach the optimum of the brute force
                brute = block_imbalance(weights, balanced_bounds_brute(weights, p))
                rec["optimal"] = bool(np.isclose(rec["predicted_imbalance"], brute))
            records.append(rec)
            if p == max(args.ranks):
                plot_details(res, f"bench_balanced_{split}_{tag(args)}_p{p}")

    if world.Get_rank() == 0:
        print(f"\n{'p':>3} {'even ms':>8} {'bal. ms':>8} {'gain':>6} "
              f"{'imb. even':>10} {'imb. bal.':>10} {'predicted':>10}  bounds")
        for ev, ba in zip(records[1::2], records[2::2]):
            print(f"{ev['p']:>3} {ev['time_per_step'] * 1e3:>8.2f} "
                  f"{ba['time_per_step'] * 1e3:>8.2f} "
                  f"{ev['time_per_step'] / ba['time_per_step']:>5.2f}x "
                  f"{ev['imbalance']:>10.2f} {ba['imbalance']:>10.2f} "
                  f"{ev['predicted_imbalance']:>4.2f}/{ba['predicted_imbalance']:<5.2f}"
                  f"  {ba['bounds']}")
    return records


def plot_balanced(records: list[dict], name: str):
    """Speedup and imbalance of the even vs. the balanced split, vs. p."""
    t1 = records[0]["time_per_step"]  # p = 1, same for both splits
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, (split, marker) in enumerate([("even", "o"), ("balanced", "s")]):
        recs = [records[0]] + [r for r in records[1:] if r["split"] == split]
        p = np.array([r["p"] for r in recs])
        t = np.array([r["time_per_step"] for r in recs])
        axes[0].plot(p, t1 / t, f"C{i}", marker=marker, label=f"{split} split")
        axes[1].plot(p, [r["imbalance"] for r in recs], f"C{i}", marker=marker,
                     label=f"{split}, measured")
        axes[1].plot(p, [r["predicted_imbalance"] for r in recs], f"C{i}", ls=":",
                     marker=marker, mfc="none", label=f"{split}, predicted from weights")
    axes[0].plot(p, p, "k--", lw=1, label="ideal")
    axes[0].set(xlabel="MPI ranks $p$", ylabel="speedup vs. 1 rank", xticks=p,
                title=f"Nx = {records[0]['Nx']}, {records[0]['n_domains']} domains")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1].set(xlabel="MPI ranks $p$", ylabel="max busy / mean busy", xticks=p,
                ylim=(0.95, None), title="Load imbalance")
    for ax in axes:
        ax.legend(fontsize=8)
    print(f"Saved {P.save_fig(fig, f'bench_balanced_{name}')}")


def dynamic(args) -> list[dict]:
    """
    Even static split vs. dynamic load balancing (balancer.Balancer), for every p.

    The dynamic runs start from the even split and repartition every `every` steps
    (one run per value of --every). p = 1 is the reference for the speedup and the
    bit-identity check, the subdomains only change their owner, never the result.
    """
    n_domains = args.n_domains or 2 * max(args.ranks)
    ref = run(1, args.Nx, n_domains, args)
    records = []
    if ref is not None:
        rec = summary(ref, 1, args.Nx, n_domains)
        rec.update(split="even", identical=True)
        records.append(rec)

    for p in [p for p in args.ranks if p > 1]:
        variants = {"even": None}
        variants.update({f"dynamic {e}": dict(every=e, weights=args.weights,
                                              min_gain=args.min_gain)
                         for e in args.every})
        for split, balance in variants.items():
            res = run(p, args.Nx, n_domains, args, balance=balance)
            if res is None:
                continue
            rec = summary(res, p, args.Nx, n_domains)
            rec.update(split=split,
                       identical=bool(np.array_equal(res.f(), ref.f())
                                      and res.rank_hist == ref.rank_hist))
            records.append(rec)
            if balance and p == max(args.ranks):
                name = f"bench_dynamic_reassign_{tag(args)}_p{p}_every{balance['every']}"
                print(f"Saved {P.save_fig(P.plot_reassignment(res), name)}")

    if world.Get_rank() == 0:
        print(f"\n{'p':>3} {'variant':>12} {'ms/step':>8} {'gain':>6} {'imbalance':>10} "
              f"{'changes':>8} {'moved':>6} {'rebal.+wait':>11}  final bounds")
        even = {}
        for rec in records[1:]:
            even.setdefault(rec["p"], rec["time_per_step"])
            bal = rec["balancer"] or {}
            share = max(rec["rebalance"]) / rec["runtime"] * 100 if bal else 0.0
            final = bal["history"][-1][1] if bal else "-"
            print(f"{rec['p']:>3} {rec['split']:>12} {rec['time_per_step'] * 1e3:>8.2f} "
                  f"{even[rec['p']] / rec['time_per_step']:>5.2f}x "
                  f"{rec['imbalance']:>10.2f} {bal.get('n_changes', '-'):>8} "
                  f"{bal.get('n_moved', '-'):>6} {share:>10.1f}%  {final}")
    return records


def plot_dynamic(records: list[dict], name: str):
    """Speedup and imbalance of the even split vs. dynamic balancing, and the owner of
    every subdomain over time (largest p, most frequent rebalancing)."""
    t1 = records[0]["time_per_step"]  # p = 1, the same for all variants
    splits = list(dict.fromkeys(r["split"] for r in records[1:]))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    markers = ["o", "s", "^", "D", "v"]
    for i, split in enumerate(splits):
        recs = [records[0]] + [r for r in records[1:] if r["split"] == split]
        p = np.array([r["p"] for r in recs])
        t = np.array([r["time_per_step"] for r in recs])
        label = "even split (static)" if split == "even" else \
            f"dynamic, every {split.split()[1]} steps"
        axes[0].plot(p, t1 / t, f"C{i}", marker=markers[i % 5], label=label)
        axes[1].plot(p, [r["imbalance"] for r in recs], f"C{i}", marker=markers[i % 5],
                     label=label)
    axes[0].plot(p, p, "k--", lw=1, label="ideal")
    axes[0].set(xlabel="MPI ranks $p$", ylabel="speedup vs. 1 rank", xticks=p,
                title=f"Nx = {records[0]['Nx']}, {records[0]['n_domains']} domains")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1].set(xlabel="MPI ranks $p$", ylabel="max busy / mean busy", xticks=p,
                ylim=(0.95, None), title="Load imbalance")

    dyn = [r for r in records if r.get("balancer")]
    if dyn:
        p_max = max(r["p"] for r in dyn)
        rec = min((r for r in dyn if r["p"] == p_max), key=lambda r: r["balancer"]["every"])
        hist = rec["balancer"]["history"]
        t = [h[0] for h in hist] + [rec["t_hist"][-1]]
        bounds = np.array([h[1] for h in hist] + [hist[-1][1]])
        for k in range(1, p_max):  # inner bounds, the outer ones are 0 and n
            axes[2].step(t, bounds[:, k], where="post", color="C0", lw=1.5)
        axes[2].set(xlabel="$t$", ylabel="subdomain index", ylim=(0, rec["n_domains"]),
                    title=f"Rank boundaries over time, p = {p_max}, every "
                          f"{rec['balancer']['every']} steps "
                          f"({rec['balancer']['n_moved']} moves)")
    else:
        axes[2].set_visible(False)
    for ax in axes[:2]:
        ax.legend(fontsize=8)
    print(f"Saved {P.save_fig(fig, f'bench_dynamic_{name}')}")


def plot_asymmetric(records: list[dict], name: str):
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
    axes[1].set(xlabel="MPI rank", ylabel="busy time (s)", ylim=(0, None), xticks=ranks,
                xticklabels=[f"P{i}" for i in ranks], title="Busy time per rank")
    for ax in axes:
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{records[0]['n_domains']} domains on {records[0]['p']} ranks, "
                 f"Nx = {records[0]['Nx']}, {name}")
    print(f"Saved {P.save_fig(fig, f'bench_asymmetric_{name}')}")

def plot_summary(experiment: str, records: list[dict], name: str):
    """Scaling plots of one experiment, name is the file tag, e.g. 'beam_Nx512'."""
    if experiment == "asymmetric":
        plot_asymmetric(records, name)
        return
    if experiment == "balanced":
        plot_balanced(records, name)
        return
    if experiment == "dynamic":
        plot_dynamic(records, name)
        return

    p = np.array([rec["p"] for rec in records])
    t = np.array([rec["time_per_step"] for rec in records]) * 1e3
    imbalance = [rec["imbalance"] for rec in records]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if experiment == "strong":
        speedup = t[0] / t
        axes[0].plot(p, speedup, "o-", label="vs. same decomposition on 1 rank")
        if records[0].get("speedup_vs_single") is not None:
            speedup_single = np.array([rec["speedup_vs_single"] for rec in records])
            axes[0].plot(p, speedup_single, "s--", label="vs. 1 domain on 1 rank")
        axes[0].plot(p, p / p[0], "k--", lw=1, label="ideal")
        axes[0].set(xlabel="MPI ranks $p$", ylabel="speedup", xticks=p,
                    title=f"Strong scaling, Nx = {records[0]['Nx']}, "
                          f"{records[0]['n_domains']} domains")
        # only vs. the same decomposition: vs. 1 domain it would be > 100 %
        axes[1].plot(p, speedup / (p / p[0]) * 100, "o-",
                     label="vs. same decomposition on 1 rank")
        axes[1].set(xlabel="MPI ranks $p$", ylabel="parallel efficiency (%)",
                    xticks=p, ylim=(0, 110), title="Efficiency")
    elif experiment == "weak":
        axes[0].plot(p, t, "o-", label="measured")
        axes[0].axhline(t[0], color="k", ls="--", lw=1, label="ideal")
        axes[0].set(xlabel="MPI ranks $p$", ylabel="time per step (ms)", xticks=p,
                    ylim=(0, None), title=f"Weak scaling, Nx = {records[0]['Nx']} per rank")
        axes[1].plot(p, imbalance, "o-")
        axes[1].set(xlabel="MPI ranks $p$", ylabel="max busy / mean busy",
                    xticks=p, title="Load imbalance")
    else:
        n = np.array([rec["n_domains"] for rec in records])
        axes[0].plot(n, t, "o-")
        axes[0].set(xlabel="number of domains", ylabel="time per step (ms)", xticks=n,
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
    print(f"Saved {P.save_fig(fig, f'bench_{experiment}_{name}')}")


def plot_compare(Nx: int):
    """Imbalance and balance E(t) of the largest p of all strong_<inflow>_Nx<Nx>.json."""
    print(f"\n{'inflow':>12} {'p':>3} {'busy imbalance':>15} {'busy E':>7} {'speedup':>8}"
          f"  final ranks")
    fig, ax = plt.subplots(figsize=(9, 5))
    fig_E, ax_E = plt.subplots(figsize=(9, 5))
    for i, inflow in enumerate(INFLOWS):
        path = DATA_DIR / f"strong_{inflow}_Nx{Nx}.json"
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
    print(f"Saved {P.save_fig(fig, f'bench_compare_inflows_Nx{Nx}')}")

    ax_E.axhline(1.0, color="k", ls="--", lw=1)
    ax_E.set(xlabel="$t$", ylabel=r"$\mathcal{E}(t)$ = min / max of the rank load",
             ylim=(0, 1.05),
             title="Load balance over time (from the low-rank r, not measured time)")
    ax_E.legend(fontsize=8, loc="lower right")
    print(f"Saved {P.save_fig(fig_E, f'bench_compare_balance_Nx{Nx}')}")


def print_table(records: list[dict]):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment",
                        choices=["strong", "weak", "domains", "asymmetric", "balanced",
                                 "dynamic", "plot"])
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
                        help="strong: total (default max ranks), weak: per rank, "
                             "balanced/dynamic: total (default 2 * max ranks)")
    parser.add_argument("--weights", choices=["measured", "model"], default="measured",
                        help="balanced: busy time per subdomain of the 1-rank run, "
                             "dynamic: since the last rebalance, "
                             "or the cost model (Nx + Nmu) r^2 + r^3")
    parser.add_argument("--min_gain", type=float, default=0.05,
                        help="dynamic: migrate only if the slowest rank gets this "
                             "fraction faster (hysteresis against flip-flopping)")
    parser.add_argument("--every", type=int, nargs="+", default=[50],
                        help="dynamic: rebalance every ... steps, one run per value")
    parser.add_argument("--domains", type=int, nargs="+", help="for 'domains'")
    args = parser.parse_args()

    if args.experiment == "plot":
        if args.plot_experiment == "compare":
            plot_compare(args.Nx)
        else:
            path = DATA_DIR / f"{args.plot_experiment}_{tag(args)}.json"
            plot_summary(args.plot_experiment, json.loads(path.read_text())["records"],
                         tag(args))
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
                      "asymmetric": asymmetric, "balanced": balanced,
                      "dynamic": dynamic}[args.experiment]
        records = experiment(args)

        if world.Get_rank() == 0:
            print_table(records)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            path = DATA_DIR / f"{args.experiment}_{tag(args)}.json"
            path.write_text(json.dumps(dict(system=info, args=vars(args), records=records),
                                       indent=1))
            print(f"Saved {path}")
            plot_summary(args.experiment, records, tag(args))