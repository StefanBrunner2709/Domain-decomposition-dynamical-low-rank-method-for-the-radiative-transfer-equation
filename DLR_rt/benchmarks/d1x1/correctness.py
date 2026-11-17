"""
Correctness of the 1x1d domain decomposition, from the original solver to MPI.

    mpirun -n 8 python -m DLR_rt.benchmarks.d1x1.correctness           (A - D)
    python -m DLR_rt.benchmarks.d1x1.correctness plot                  (only plot)
    mpirun -n 4 python -m DLR_rt.benchmarks.d1x1.correctness compare 8 (one case in
                                                     detail, 8 subdomains on 4 ranks)

run, the chain of comparisons:
  A) original solver (examples/d1x1/dlr_inflow) vs. domains.integrate_serial with
     1 subdomain: the restructured code has to be bit-identical
  B) serial DD with n subdomains vs. 1 subdomain: the error of the decomposition,
     against n, the tolerance and the time step
  C) MPI on p processes vs. serial DD: has to be bit-identical
  D) without the exchange vs. with it: the size of an exchange bug (control)
compare: one case, f, the pointwise difference MPI vs. serial, the errors over time
     of MPI, the control and the DD, and the rank histories.

Results: data/benchmarks/correctness.json, plots/benchmark_correctness/,
         plots/compare_serial_mpi_<p>ranks_<n>domains.png / .npz
"""
from DLR_rt.benchmarks.d1x1 import common as C  # first, sets the BLAS threads

import argparse  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from mpi4py import MPI  # noqa: E402

from DLR_rt.src.d1x1 import plotting as P  # noqa: E402
from DLR_rt.src.d1x1.domains import RunResult, setup_domains  # noqa: E402

world = MPI.COMM_WORLD

BASE = dict(Nx=256, Nmu=64, r=5, seed=0, t_f=0.5, cfl=0.25, tol=1e-5)
N_DOMAINS = [2, 4, 8, 16]
TOLS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
CFLS = [0.4, 0.2, 0.1, 0.05]
MPI_CASES = [(2, 2), (4, 4), (4, 8), (3, 8), (8, 8), (8, 16)]  # (processes, domains)
N_SWEEP = 4  # subdomains for the tol and dt sweeps
TOL_DT_LOW = 1e-8  # second dt sweep, where the tol sweep has reached its floor
# compare: longer and finer in mu, f every 10 steps for the error over time
COMPARE = dict(Nx=256, Nmu=128, r=5, seed=0, t_f=1.0, cfl=0.257,
               tol=1e-5)  # dt = cfl / (Nx + 1) = 1e-3


def serial(cfg, n_domains, **kw) -> RunResult:
    """Serial DD, one run without warm-up."""
    Nx = cfg["Nx"]
    return C.run_serial(Nx, n_domains, **{k: v for k, v in cfg.items() if k != "Nx"},
                        warmup=False, **kw)


def mpi(cfg, p, n_domains, **kw) -> RunResult | None:
    """Collective on all ranks: runs on the first p, result on rank 0."""
    Nx = cfg["Nx"]
    return C.run_mpi(p, Nx, n_domains, **{k: v for k, v in cfg.items() if k != "Nx"},
                     warmup=False, comm=world, **kw)


def original(cfg):
    """The original single-domain solver dlr_inflow, returns f and the rank history."""
    from DLR_rt.examples.d1x1 import dlr_inflow

    grids, lrs, rngs = setup_domains(cfg["Nx"], cfg["Nmu"], cfg["r"], 1, cfg["seed"])
    lr, _, ranks, _, _ = dlr_inflow.integrate(
        lrs[0], grids[0], cfg["t_f"], C.dt_of(cfg["Nx"], cfg["cfl"]),
        tol_sing_val=cfg["tol"], drop_tol=cfg["tol"], rng=rngs[0])
    return lr.U @ lr.S @ lr.V.T, ranks


def rel_err(a, b) -> float:
    """Relative L2 error of a against b."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def compare(a_f, b_f, a_ranks=None, b_ranks=None) -> dict:
    out = dict(identical=bool(np.array_equal(a_f, b_f)),
               max_abs=float(np.max(np.abs(a_f - b_f))),
               rel_l2=rel_err(a_f, b_f))
    if a_ranks is not None:
        out["same_ranks"] = [list(r) for r in a_ranks] == [list(r) for r in b_ranks]
    return out


### Chain A - D

def chain_serial() -> dict:
    """A, B and D, only on rank 0."""
    out = {}
    one = serial(BASE, 1, snap_every=10)

    # A: original solver against the restructured code with one subdomain
    f_orig, ranks_orig = original(BASE)
    out["A"] = compare(one.f(), f_orig, one.rank_hist, [ranks_orig])

    # B1: DD error against the number of subdomains
    out["B_domains"] = []
    for n in N_DOMAINS:
        dd = serial(BASE, n, snap_every=10)
        rec = dict(n_domains=n, rel_l2=rel_err(dd.f(), one.f()),
                   rank_max=int(np.max(dd.rank_hist)))
        if n == N_SWEEP:  # error over time for the plot
            rec["t"] = dd.snap_t
            rec["rel_l2_t"] = [rel_err(a, b) if np.linalg.norm(b) > 0 else None
                               for a, b in zip(dd.snap_f, one.snap_f)]
        out["B_domains"].append(rec)

    # B2: against the tolerance, B3: against the time step
    out["B_tol"] = [dict(tol=tol, rel_l2=rel_err(serial(cfg, N_SWEEP).f(),
                                                 serial(cfg, 1).f()))
                    for tol in TOLS for cfg in [{**BASE, "tol": tol}]]
    # At the default tol the truncation per step dominates (more steps, more error),
    # at TOL_DT_LOW the rank truncation is negligible and the coupling error remains
    out["B_dt"] = [dict(cfl=c, dt=C.dt_of(cfg["Nx"], c), tol=tol,
                        rel_l2=rel_err(serial(cfg, N_SWEEP).f(), serial(cfg, 1).f()))
                   for tol in [BASE["tol"], TOL_DT_LOW] for c in CFLS
                   for cfg in [{**BASE, "cfl": c, "tol": tol}]]

    # D: negative control, the subdomains ignore each other
    dd = serial(BASE, N_SWEEP)
    ctrl = serial(BASE, N_SWEEP, use_neighbours=False)
    out["D"] = dict(n_domains=N_SWEEP, rel_l2=rel_err(ctrl.f(), dd.f()))
    return out


def chain_mpi() -> list[dict]:
    """C: all ranks call this together, comparison on rank 0."""
    records = []
    for p, n in MPI_CASES:
        if p > world.Get_size():
            continue
        res = mpi(BASE, p, n)
        if world.Get_rank() == 0:
            ser = serial(BASE, n)
            records.append(dict(p=p, n_domains=n, **compare(
                res.f(), ser.f(), res.rank_hist, ser.rank_hist)))
    return records


def print_table(out: dict):
    a = out["A"]
    print(f"\nA  dlr_inflow vs domains (1 domain): identical={a['identical']}, "
          f"same ranks={a['same_ranks']}, max abs={a['max_abs']:.1e}")
    print("\nB  DD vs 1 domain")
    for r in out["B_domains"]:
        print(f"   n={r['n_domains']:3d}  rel L2 = {r['rel_l2']:.2e}")
    for r in out["B_tol"]:
        print(f"   tol={r['tol']:.0e} (n={N_SWEEP})  rel L2 = {r['rel_l2']:.2e}")
    for r in out["B_dt"]:
        print(f"   dt={r['dt']:.1e} tol={r['tol']:.0e} (n={N_SWEEP})  "
              f"rel L2 = {r['rel_l2']:.2e}")
    print("\nC  MPI vs serial DD")
    for r in out["C"]:
        print(f"   p={r['p']} n={r['n_domains']:3d}  identical={r['identical']}  "
              f"same ranks={r['same_ranks']}  max abs={r['max_abs']:.1e}")
    print(f"\nD  control without exchange: rel L2 = {out['D']['rel_l2']:.2e}")


def plot(out: dict, base: dict = BASE):
    """DD error vs. number of subdomains and over time (relative L2 vs. 1 domain)."""
    with plt.rc_context({"font.size": 13}):
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        settings = (f"Nx = {base['Nx']}, N$_\\mu$ = {base['Nmu']}, "
                    f"tol = {base['tol']:.0e}, $t_f$ = {base['t_f']}")

        # (a) Error at t_f against the number of subdomains
        b = out["B_domains"]
        n = [r["n_domains"] for r in b]
        err = [r["rel_l2"] for r in b]
        ax[0].plot(n, err, "o-", ms=7, lw=2)
        ax[0].set_xscale("log", base=2)
        ax[0].set_xticks(n, [str(k) for k in n])  # 2, 4, 8, 16 instead of 2^1, ...
        ax[0].minorticks_off()
        ax[0].set_yscale("log")
        # One decade around the data, so the (small) growth with n is not exaggerated
        lo = 10 ** np.floor(np.log10(min(err)))
        ax[0].set_ylim(lo, 10 * lo)
        ax[0].set(xlabel="number of subdomains", ylabel="rel. $L^2$ error vs. 1 domain",
                  title=f"(a) DD error at $t_f$ = {base['t_f']}")

        # (b) Error over time for N_SWEEP subdomains (t = 0 is exactly 0, left out)
        rec = next(r for r in b if "t" in r)
        t = [ti for ti, e in zip(rec["t"], rec["rel_l2_t"]) if e]
        e = [e for e in rec["rel_l2_t"] if e]
        ax[1].semilogy(t, e, lw=2)
        ax[1].set_xlim(0, base["t_f"])
        ax[1].set(xlabel="$t$", ylabel="rel. $L^2$ error vs. 1 domain",
                  title=f"(b) DD error over time, {rec['n_domains']} subdomains")

        for a in ax:
            a.grid(True, which="both", alpha=0.3)
        fig.suptitle(settings, fontsize=12)
        fig.tight_layout()
        out_dir = P.PLOTS_DIR / "benchmark_correctness"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "correctness.png", dpi=200)
        plt.close(fig)
        print(f"Saved {out_dir / 'correctness.png'}")


### compare: one case in detail

def errors(a: RunResult, b: RunResult) -> dict:
    """Errors of a against the reference b per snapshot."""
    assert np.allclose(a.snap_t, b.snap_t), "Snapshots at different times"
    diff = np.stack(a.snap_f) - np.stack(b.snap_f)
    # Uniform grid, the weights dx dmu cancel in the relative L2 error
    norm_diff = np.linalg.norm(diff, axis=(1, 2))
    norm_ref = np.linalg.norm(np.stack(b.snap_f), axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):  # f(0) = 0
        rel = np.where(norm_ref > 0, norm_diff / norm_ref, np.nan)
    return dict(max_abs=np.max(np.abs(diff), axis=(1, 2)), rel_l2=rel)


def report(name: str, a: RunResult, b: RunResult) -> dict:
    """Print the comparison of a against the reference b."""
    err = errors(a, b)
    print(f"\n### {name}")
    print(f"Bit-identical:         {np.array_equal(a.f(), b.f())}")
    print(f"Max abs error at t_f:  {err['max_abs'][-1]:.2e}")
    print(f"Rel. L2 error at t_f:  {err['rel_l2'][-1]:.2e}")
    if a.n_domains == b.n_domains:
        same_rank = a.rank_hist == b.rank_hist
        print(f"Same rank history:     {same_rank}")
        if not same_rank:
            n = next(i for i in range(len(a.time))
                     if any(ra[i] != rb[i] for ra, rb in zip(a.rank_hist, b.rank_hist)))
            print(f"Rank history differs first at step {n}, t = {a.time[n]:.3f}")
    return err


def plot_comparison(ser: RunResult, mpi_res: RunResult, single: RunResult,
                    err_mpi, err_ctrl, err_dd, name: str):
    """(a) MPI solution, (b) |MPI - serial|, (c) errors over time, (d) rank histories."""
    p = len(set(mpi_res.owner))
    g = ser.grids
    extent = [g[0].X[0], g[-1].X[-1], g[0].MU[0], g[0].MU[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def image(ax, f, title, **kw):
        im = ax.imshow(f.T, extent=extent, origin="lower", aspect=0.5, **kw)
        for x in P.interfaces(ser):
            ax.axvline(x, color="w", ls="--", lw=1)
        ax.set(xlabel="$x$", ylabel=r"$\mu$", title=title)
        fig.colorbar(im, ax=ax)

    image(axes[0, 0], mpi_res.f(), f"(a) MPI, {p} processes, {mpi_res.n_domains} "
                                   f"domains, $t=${mpi_res.time[-1]:g}")
    diff = np.abs(mpi_res.f() - ser.f())
    image(axes[0, 1], np.maximum(diff, P.FLOOR),
          r"(b) $|f_{MPI} - f_{serial}|$ at $t_f$"
          + ("" if diff.any() else " (bit-identical)"),
          norm=LogNorm(vmin=P.FLOOR, vmax=max(diff.max(), 1e-15)))

    t = np.array(ser.snap_t)
    P.plot_error_over_time(axes[1, 0], t, {
        "MPI vs serial": err_mpi["rel_l2"],
        "MPI without exchange vs serial (control)": err_ctrl["rel_l2"],
        "DD vs one domain (DD error)": err_dd["rel_l2"]})
    axes[1, 0].set_title("(c) Relative error over time")

    # (d) serial wide and light, MPI dashed on top. The problem is symmetric,
    # mirrored subdomains can lie on top of each other
    ax = axes[1, 1]
    for i, (r_ser, r_mpi) in enumerate(zip(ser.rank_hist, mpi_res.rank_hist)):
        ax.plot(ser.time, r_ser, color=f"C{i}", lw=4, alpha=0.4,
                label=f"serial, domain {i}")
        ax.plot(mpi_res.time, r_mpi, "--", color=f"C{i}", lw=1.5,
                label=f"MPI, domain {i} (process P{mpi_res.owner[i]})")
    ax.plot(single.time, single.rank_hist[0], ":", color="grey", label="one domain")
    ax.set(xlabel="$t$", ylabel="low rank $r(t)$", title="(d) Low rank history")
    ax.legend(fontsize=8)
    fig.tight_layout()
    print(f"Saved {P.save_fig(fig, name)}")

    # All curves, so the plots can be redone without running again
    np.savez(P.PLOTS_DIR / f"{name}.npz",
             t=t, f_ser=ser.f(), f_mpi=mpi_res.f(),
             rel_mpi=err_mpi["rel_l2"], rel_ctrl=err_ctrl["rel_l2"],
             rel_dd=err_dd["rel_l2"], max_abs_mpi=err_mpi["max_abs"],
             time=np.array(ser.time), ranks_ser=np.array(ser.rank_hist),
             ranks_mpi=np.array(mpi_res.rank_hist), owner=np.array(mpi_res.owner),
             time_single=np.array(single.time),
             ranks_single=np.array(single.rank_hist[0]))


def compare_case(n_domains: int):
    """MPI on all ranks vs. serial, control without exchange, DD vs. 1 subdomain."""
    p = world.Get_size()
    mpi_res = mpi(COMPARE, p, n_domains, snap_every=10)
    ctrl = mpi(COMPARE, p, n_domains, snap_every=10, use_neighbours=False)
    if world.Get_rank() != 0:
        return
    ser = serial(COMPARE, n_domains, snap_every=10)
    single = serial(COMPARE, 1, snap_every=10)

    err_mpi = report("MPI vs serial", mpi_res, ser)
    err_ctrl = report("MPI without exchange vs serial (control)", ctrl, ser)
    err_dd = report("DD vs one domain", ser, single)
    plot_comparison(ser, mpi_res, single, err_mpi, err_ctrl, err_dd,
                    f"compare_serial_mpi_{p}ranks_{n_domains}domains")

    # Fails loudly if a change breaks the MPI version
    assert mpi_res.rank_hist == ser.rank_hist, "MPI changes the rank history!"
    assert np.nanmax(err_mpi["rel_l2"]) < 1e-12, "MPI changes the result!"
    print("\nTest passed")


if __name__ == "__main__":
    ps = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ps.add_argument("mode", nargs="?", default="run", choices=["run", "plot", "compare"])
    ps.add_argument("n_domains", nargs="?", type=int,
                    help="compare: number of subdomains, default mpirun -n")
    args = ps.parse_args()

    if args.mode == "plot":
        content = C.load_json("correctness")
        plot(content["results"], content.get("base", BASE))
    elif args.mode == "compare":
        compare_case(args.n_domains or world.Get_size())
    else:
        info = C.start_mpi(world)
        out = chain_serial() if world.Get_rank() == 0 else None
        out_c = chain_mpi()  # collective, the other ranks wait here for rank 0
        if world.Get_rank() == 0:
            out["C"] = out_c
            print_table(out)
            path = C.save_json("correctness", info, base=BASE, results=out)
            print(f"Saved {path}")
            plot(out)
            # A and C have to be exact
            assert out["A"]["identical"] and out["A"]["same_ranks"], "Refactor changed f!"
            assert all(r["identical"] and r["same_ranks"] for r in out["C"]), \
                "MPI changed f!"
            print("\nA and C passed")
