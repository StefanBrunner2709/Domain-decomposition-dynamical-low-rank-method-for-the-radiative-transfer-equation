"""
Load balancing of the 1x1d domain decomposition: n_domains > p subdomains are
distributed onto p MPI processes in contiguous blocks.

    mpirun -n 8 python -m DLR_rt.benchmarks.d1x1.balancing balanced --n_domains 32
    mpirun -n 8 python -m DLR_rt.benchmarks.d1x1.balancing dynamic --n_domains 32 \
        --inflow beam --every 10 50 200
    python -m DLR_rt.benchmarks.d1x1.balancing plot dynamic --inflow beam \
        --n_domains 32                (only plot, the same --inflow/--Nx/--weights/
                                       --n_domains/--suffix as the run)

balanced: static. Even split vs. balanced split (partition.balanced_bounds) for
          every p, with weights from the 1-process run: its measured busy time per
          subdomain, or the cost model with its ranks (--weights model).
dynamic:  the split is recomputed during the run every --every steps and the
          subdomains migrate (balancer.Balancer), vs. the static even split.

Only the owner of a subdomain changes, so every run has to be bit-identical to the
1-process run (column 'identical').
Results: data/benchmarks/dynamic_beam_Nx512_measured_n32.json, plots/bench_dynamic_...
Clean timings: run alone, mpirun --bind-to core --map-by core ... --repeat 3
"""
from DLR_rt.benchmarks.d1x1 import common as C  # first, sets the BLAS threads

import argparse  # noqa: E402
import math  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from DLR_rt.src.d1x1 import plotting as P  # noqa: E402
from DLR_rt.src.d1x1.inflow import INFLOWS  # noqa: E402
from DLR_rt.src.d1x1.partition import (  # noqa: E402
    balanced_bounds,
    balanced_bounds_brute,
    block_imbalance,
    even_bounds,
    measured_weights,
    model_weights,
)


def case(args) -> dict:
    """Keyword arguments of C.run_mpi that are the same for all runs."""
    return dict(Nmu=args.Nmu, r=args.r, t_f=args.t_f, cfl=args.cfl, tol=args.tol,
                seed=args.seed, inflow=args.inflow, repeat=args.repeat)


def tag(args) -> str:
    return C.tag(args.inflow, args.Nx, args.suffix, weights=args.weights,
                 n_domains=args.n_domains)


### Experiments, collective on all ranks, the records only on rank 0

def balanced(args, world) -> list[dict]:
    """
    Even vs. balanced split of the same decomposition, for every p.

    1. p = 1: reference for the speedup and the bit-identity check. Its busy time
       per subdomain gives the weights, the cost of a subdomain does not depend on
       which process computes it (--weights model: cost model with its ranks).
    2. For every p > 1: even split and balanced split (smallest maximum block of
       the weights). The result is bit-identical, only the distribution differs.
    """
    n_domains = args.n_domains or 2 * max(args.ranks)
    ref = C.run_mpi(1, args.Nx, n_domains, **case(args))
    weights = None
    if ref is not None:
        weights = (measured_weights(ref.timings) if args.weights == "measured"
                   else model_weights(ref.grids, ref.rank_hist))
    weights = world.bcast(weights, root=0)  # same weights, so same bounds everywhere

    records = []
    if ref is not None:
        rec = C.summary(ref, 1, args.Nx, n_domains)
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
            res = C.run_mpi(p, args.Nx, n_domains, **case(args), bounds=bounds)
            if res is None:
                continue
            rec = C.summary(res, p, args.Nx, n_domains)
            rec.update(split=split, bounds=bounds,
                       predicted_imbalance=block_imbalance(weights, bounds),
                       identical=C.identical(res, ref))
            if split == "balanced" and math.comb(n_domains - 1, p - 1) <= 200_000:
                # The bisection has to reach the optimum of the brute force
                brute = block_imbalance(weights, balanced_bounds_brute(weights, p))
                rec["optimal"] = bool(np.isclose(rec["predicted_imbalance"], brute))
            records.append(rec)
            if p == max(args.ranks):
                C.plot_details(res, f"bench_balanced_{split}_{tag(args)}_p{p}")

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


def dynamic(args, world) -> list[dict]:
    """
    Even static split vs. dynamic load balancing (balancer.Balancer), for every p.

    The dynamic runs start from the even split and repartition every `every` steps
    (one run per value of --every). p = 1 is the reference for the speedup and the
    bit-identity check, the subdomains only change their owner, never the result.
    """
    n_domains = args.n_domains or 2 * max(args.ranks)
    ref = C.run_mpi(1, args.Nx, n_domains, **case(args))
    records = []
    if ref is not None:
        rec = C.summary(ref, 1, args.Nx, n_domains)
        rec.update(split="even", identical=True)
        records.append(rec)

    for p in [p for p in args.ranks if p > 1]:
        variants = {"even": None}
        variants.update({f"dynamic {e}": dict(every=e, weights=args.weights,
                                              min_gain=args.min_gain)
                         for e in args.every})
        for split, balance in variants.items():
            res = C.run_mpi(p, args.Nx, n_domains, **case(args), balance=balance)
            if res is None:
                continue
            rec = C.summary(res, p, args.Nx, n_domains)
            rec.update(split=split, identical=C.identical(res, ref))
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


EXPERIMENTS = {"balanced": balanced, "dynamic": dynamic}


### Plots from the records, also for older result files

def plot_balanced(records: list[dict], name: str):
    """Speedup and imbalance of the even vs. the balanced split, vs. p."""
    t1 = records[0]["time_per_step"]  # p = 1, the same for both splits
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
    axes[0].set(xlabel="MPI processes $p$", ylabel="speedup vs. 1 process", xticks=p,
                title=f"Nx = {records[0]['Nx']}, {records[0]['n_domains']} domains")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1].set(xlabel="MPI processes $p$", ylabel="max busy / mean busy", xticks=p,
                ylim=(0.95, None), title="Load imbalance")
    for ax in axes:
        ax.legend(fontsize=8)
    print(f"Saved {P.save_fig(fig, f'bench_balanced_{name}')}")


def plot_dynamic(records: list[dict], name: str):
    """
    Speedup and imbalance of the even split vs. dynamic balancing, and the process
    boundaries over time (largest p, most frequent rebalancing).
    """
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
    axes[0].set(xlabel="MPI processes $p$", ylabel="speedup vs. 1 process", xticks=p,
                title=f"Nx = {records[0]['Nx']}, {records[0]['n_domains']} domains")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1].set(xlabel="MPI processes $p$", ylabel="max busy / mean busy", xticks=p,
                ylim=(0.95, None), title="Load imbalance")

    dyn = [r for r in records if r.get("balancer")]
    if dyn:
        p_max = max(r["p"] for r in dyn)
        rec = min((r for r in dyn if r["p"] == p_max), key=lambda r: r["balancer"]["every"])
        t, bounds = P.bounds_steps(rec["balancer"]["history"], rec["t_hist"][-1])
        for k in range(1, p_max):  # inner bounds, the outer ones are 0 and n
            axes[2].step(t, bounds[:, k], where="post", color="C0", lw=1.5)
        axes[2].set(xlabel="$t$", ylabel="subdomain index", ylim=(0, rec["n_domains"]),
                    title=f"Process boundaries over time, p = {p_max}, every "
                          f"{rec['balancer']['every']} steps "
                          f"({rec['balancer']['n_moved']} moves)")
    else:
        axes[2].set_visible(False)
    for ax in axes[:2]:
        ax.legend(fontsize=8)
    print(f"Saved {P.save_fig(fig, f'bench_dynamic_{name}')}")


PLOTS = {"balanced": plot_balanced, "dynamic": plot_dynamic}


def parser() -> argparse.ArgumentParser:
    ps = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ps.add_argument("experiment", choices=[*EXPERIMENTS, "plot"])
    ps.add_argument("plot_experiment", nargs="?", choices=list(EXPERIMENTS),
                    help="for 'plot': which one")
    ps.add_argument("--Nx", type=int, default=512)
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
    ps.add_argument("--n_domains", type=int, help="total, default 2 * max p")
    ps.add_argument("--weights", choices=["measured", "model"], default="measured",
                    help="balanced: busy time per subdomain of the 1-process run, "
                         "dynamic: since the last rebalance, "
                         "or the cost model (Nx + Nmu) r^2 + r^3")
    ps.add_argument("--every", type=int, nargs="+", default=[50],
                    help="dynamic: rebalance every ... steps, one run per value")
    ps.add_argument("--min_gain", type=float, default=0.05,
                    help="dynamic: migrate only if the slowest process gets this "
                         "fraction faster (hysteresis against flip-flopping)")
    ps.add_argument("--suffix", default="",
                    help="appended to all result file names (also for 'plot')")
    return ps


if __name__ == "__main__":
    args = parser().parse_args()

    if args.experiment == "plot":
        name = f"{args.plot_experiment}_{tag(args)}"
        PLOTS[args.plot_experiment](C.load_json(name)["records"], tag(args))
    else:
        from mpi4py import MPI

        world = MPI.COMM_WORLD
        size = world.Get_size()
        args.ranks = args.ranks or [p for p in (1, 2, 4, 8, 16, 32, 64) if p <= size]
        assert max(args.ranks) <= size, "Start mpirun with at least max(--ranks) ranks"

        info = C.start_mpi(world)
        records = EXPERIMENTS[args.experiment](args, world)
        if world.Get_rank() == 0:
            C.print_table(records)
            path = C.save_json(f"{args.experiment}_{tag(args)}", info, args=vars(args),
                               records=records)
            print(f"Saved {path}")
            PLOTS[args.experiment](records, tag(args))
