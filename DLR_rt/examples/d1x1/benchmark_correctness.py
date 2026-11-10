""" 
Correctness of the 1x1d domain decomposition, from the original solver to MPI

    mpirun -n 8 python -m DLR_rt.examples.d1x1.benchmark_correctness
    python -m DLR_rt.examples.d1x1.benchmark_correctness plot      (only plot)

Comparisons:

A) dlr_inflow (original solver) vs dd_1x1d, 1 domain --> see if the refactor works
B) dd_1x1d, n domains vs dd_1x1d, 1 domain --> error of the DD
C) dd_mpi, p ranks  vs dd_1x1d, n domains --> MPI exact?
D) without exchange vs with exchange --> scale of exchange bug
"""
import os

# BLAS and LAPACK to 1 thread per rank, always overwrite (shell sets OMP_NUM_THREADS)
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from DLR_rt.examples.d1x1 import dlr_inflow
from DLR_rt.src.dd_1x1d import full_f, integrate_serial, setup_domains
from DLR_rt.src.dd_mpi import domain_slice, gather_result, integrate_mpi
from DLR_rt.src.sysinfo import print_system_info, system_info

world = MPI.COMM_WORLD
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "benchmarks" / "correctness.json"
PLOT_DIR = ROOT / "plots" / "benchmark_correctness"

BASE = dict(Nx=256, Nmu=64, r0=5, seed=0, t_f=0.5, cfl=0.25, tol=1e-5)
N_DOMAINS = [2, 4, 8, 16]
TOLS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
CFLS = [0.4, 0.2, 0.1, 0.05]
MPI_CASES = [(2, 2), (4, 4), (4, 8), (3, 8), (8, 8), (8, 16)]  # (ranks, domains)
N_SWEEP = 4  # domains for the tol and dt sweeps
TOL_DT_LOW = 1e-8  # second dt sweep, where the tol sweep has reached its floor

def dt_of(cfg) -> float:
    """ 
    dt = cfl * dx with dx = 1/(Nx + 1), the same for every decomposition
    """
    return cfg["cfl"] / (cfg["Nx"] + 1)

def setup(cfg, n_domains):
    """ 
    Setup the domains
    """
    return setup_domains(cfg["Nx"], cfg["Nmu"], cfg["r0"], n_domains, cfg["seed"])

def original(cfg):
    """ 
    Original single domain solver dlr_inflow, returns f and rank history
    """
    grids, lr0s, rngs = setup(cfg, 1)
    lr, _, ranks, _, _ = dlr_inflow.integrate(
        lr0s[0], grids[0], cfg["t_f"], dt_of(cfg),
        tol_sing_val=cfg["tol"], drop_tol=cfg["tol"], rng=rngs[0])
    return full_f(lr), ranks

def serial(cfg, n_domains, **kw):
    """
    Serial domain decomposition
    """
    grids, lr0s, rngs = setup(cfg, n_domains)
    return integrate_serial(lr0s, grids, rngs, cfg["t_f"], dt_of(cfg),
                            cfg["tol"], cfg["tol"], progress=False, **kw)

def mpi(cfg, p, n_domains):
    """Collective on all ranks: runs on the first p ranks, result on rank 0."""
    rank = world.Get_rank()
    sub = world.Split(0 if rank < p else MPI.UNDEFINED, rank)
    res = None
    if sub != MPI.COMM_NULL:
        grids, lr0s, rngs = setup(cfg, n_domains)
        mine = domain_slice(n_domains, sub)
        local = integrate_mpi(lr0s[mine], grids[mine], rngs[mine], cfg["t_f"],
                              dt_of(cfg), sub, cfg["tol"], cfg["tol"], progress=False)
        res = gather_result(sub, local)
        sub.Free()
    return res

def rel_err(a,b) -> float:
    """ 
    Relative L2 error of a against b
    """
    return float(np.linalg.norm(a-b) / np.linalg.norm(b))

def compare(a_f, b_f, a_ranks=None, b_ranks=None) -> dict:
    """ 
    Compare two results
    """
    out = dict(identical=bool(np.array_equal(a_f, b_f)),
               max_abs = float(np.max(np.abs(a_f - b_f))),
               rel_l2 = rel_err(a_f, b_f))
    if a_ranks is not None:
        out["same_ranks"] = [list(r) for r in a_ranks] == [list(r) for r in b_ranks]

    return out

### Experiments
def chain_serial() -> dict:
    """A, B and D, only on rank 0."""
    out = {}
    one = serial(BASE, 1, snap_every=10)

    # A: original solver against the refactored code with one domain
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
    out["B_tol"] = [dict(tol=tol, rel_l2=rel_err(serial(cfg, N_SWEEP).f(), serial(cfg, 1).f()))
                    for tol in TOLS for cfg in [{**BASE, "tol": tol}]]
    # At the default tol the truncation per step dominates (more steps, more error),
    # at TOL_DT_LOW the rank truncation is negligible and the coupling error remains
    out["B_dt"] = [dict(cfl=c, dt=dt_of(cfg), tol=tol,
                        rel_l2=rel_err(serial(cfg, N_SWEEP).f(), serial(cfg, 1).f()))
                   for tol in [BASE["tol"], TOL_DT_LOW] for c in CFLS
                   for cfg in [{**BASE, "cfl": c, "tol": tol}]]

    # D: negative control, domains ignore each other
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
            records.append(dict(p=p, n_domains=n,
                                **compare(res.f(), ser.f(), res.rank_hist, ser.rank_hist)))
    return records


### Output
def print_table(out: dict):
    a = out["A"]
    print(f"\nA  dlr_inflow vs dd_1x1d (1 domain): identical={a['identical']}, "
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


def slope(x, y) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def plot(out: dict):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ctrl = out["D"]["rel_l2"]

    b = out["B_domains"]
    ax[0, 0].plot([r["n_domains"] for r in b], [r["rel_l2"] for r in b], "o-")
    ax[0, 0].set_xscale("log", base=2)  # only x in base 2, the error in base 10
    ax[0, 0].set_yscale("log")
    ax[0, 0].set(xlabel="number of subdomains", ylabel="rel. $L^2$ error vs 1 domain",
                 title="(a) DD error vs subdomains")

    b = out["B_tol"]
    ax[0, 1].loglog([r["tol"] for r in b], [r["rel_l2"] for r in b], "o-")
    ax[0, 1].invert_xaxis()
    ax[0, 1].set(xlabel="tolerance", ylabel="rel. $L^2$ error",
                 title=f"(b) DD error vs tolerance ({N_SWEEP} domains)")

    for tol in sorted({r.get("tol", BASE["tol"]) for r in out["B_dt"]}, reverse=True):
        b = [r for r in out["B_dt"] if r.get("tol", BASE["tol"]) == tol]
        dt, err = [r["dt"] for r in b], [r["rel_l2"] for r in b]
        ax[1, 0].loglog(dt, err, "o-", label=f"tol {tol:.0e}, slope {slope(dt, err):.2f}")
    ax[1, 0].set(xlabel=r"$\Delta t$", ylabel="rel. $L^2$ error",
                 title=f"(c) DD error vs time step ({N_SWEEP} domains)")

    rec = next(r for r in out["B_domains"] if "t" in r)
    t = [ti for ti, e in zip(rec["t"], rec["rel_l2_t"]) if e is not None]
    e = [e for e in rec["rel_l2_t"] if e is not None]
    ax[1, 1].semilogy(t, e, label=f"DD, {N_SWEEP} domains")
    ax[1, 1].set(xlabel="$t$", ylabel="rel. $L^2$ error",
                 title="(d) DD error over time")

    for a in ax.flat:
        a.axhline(ctrl, color="r", ls=":", lw=1, label="no exchange (control)")
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "correctness.png", dpi=200)
    print(f"Saved {PLOT_DIR / 'correctness.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="run", choices=["run", "plot"])
    args = parser.parse_args()

    if args.mode == "plot":
        plot(json.loads(DATA.read_text())["results"])
    else:
        info = None
        if world.Get_rank() == 0:
            info = system_info()
            print_system_info(info)
            out = chain_serial()
        out_c = chain_mpi()  # collective, the other ranks wait here for rank 0
        if world.Get_rank() == 0:
            out["C"] = out_c
            print_table(out)
            DATA.parent.mkdir(parents=True, exist_ok=True)
            DATA.write_text(json.dumps(dict(system=info, base=BASE, results=out), indent=1))
            print(f"Saved {DATA}")
            plot(out)
            # Fail if A and C must be exact
            assert out["A"]["identical"] and out["A"]["same_ranks"], "Refactor changed f!"
            assert all(r["identical"] and r["same_ranks"] for r in out["C"]), "MPI changed f!"
            print("\nA and C passed")