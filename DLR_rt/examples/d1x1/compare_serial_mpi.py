"""
Compare the MPI version (dd_mpi.integrate_mpi) against the serial version
(dd_1x1d.integrate_serial) on the same subdomains.

Two controls show the scale of a real error:
- MPI without neighbour exchange (negative control, domains ignore each other)
- serial DD against one domain without decomposition,
  this is the actual error of the domain decomposition

    mpirun -n 2 python -m DLR_rt.examples.d1x1.compare_serial_mpi       (2 domains)
    mpirun -n 4 python -m DLR_rt.examples.d1x1.compare_serial_mpi 8     (8 domains)

Results: plots/compare_serial_mpi_{p}ranks_{n}domains.png / .npz
"""
import os

# BLAS and LAPACK to 1 thread per MPI rank, has to be set before numpy is imported
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpi4py import MPI

from DLR_rt.src.dd_1x1d import RunResult, integrate_serial, setup_domains
from DLR_rt.src.dd_mpi import domain_slice, gather_result, integrate_mpi

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

### Config
SEED = 0
NX = 256
NMU = 128
R0 = 5
DT = 1e-3
T_F = 1.0
TOL = 1e-5
SNAP_EVERY = 10  # store f every 10 steps -> error over time
# Errors of exactly 0 are drawn at this value on the log scale
FLOOR = 1e-17

RUN = dict(t_f=T_F, dt=DT, tol_sing_val=TOL, drop_tol=TOL, snap_every=SNAP_EVERY)


### Runs, all return a RunResult with every subdomain (MPI: only on rank 0)
def run_mpi(n_domains, use_neighbours=True) -> RunResult | None:
    """All ranks together, result only on rank 0."""
    grids, lr0s, rngs = setup_domains(NX, NMU, R0, n_domains, SEED)
    mine = domain_slice(n_domains, comm)
    comm.Barrier()
    local = integrate_mpi(lr0s[mine], grids[mine], rngs[mine], comm=comm,
                          use_neighbours=use_neighbours, **RUN)
    return gather_result(comm, local)


def run_serial(n_domains) -> RunResult:
    """Serial DD with the same seeds, only on rank 0."""
    grids, lr0s, rngs = setup_domains(NX, NMU, R0, n_domains, SEED)
    return integrate_serial(lr0s, grids, rngs, **RUN)


### Error measures
def errors(a: RunResult, b: RunResult):
    """Errors of a against reference b per snapshot."""
    assert np.allclose(a.snap_t, b.snap_t), "Snapshots at different times"
    diff = np.stack(a.snap_f) - np.stack(b.snap_f)
    ref = np.stack(b.snap_f)
    # Uniform grid, so the weights dx*dmu cancel in the relative L2 error
    norm_diff = np.linalg.norm(diff, axis=(1, 2))
    norm_ref = np.linalg.norm(ref, axis=(1, 2))
    # f(0) = 0, the relative error is not defined there
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(norm_ref > 0, norm_diff / norm_ref, np.nan)
    return dict(max_abs=np.max(np.abs(diff), axis=(1, 2)), rel_l2=rel)


def report(name, a: RunResult, b: RunResult):
    """Print the comparison of a against reference b."""
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


### Plots
def plot_comparison(ser: RunResult, mpi: RunResult, single: RunResult,
                    err_mpi, err_ctrl, err_dd, savepath: Path, tag: str):
    grids = ser.grids
    extent = [grids[0].X[0], grids[-1].X[-1], grids[0].MU[0], grids[0].MU[-1]]
    interfaces = [0.5 * (g_a.X[-1] + g_b.X[0]) for g_a, g_b in zip(grids, grids[1:])]

    def mark_interfaces(ax):
        for x in interfaces:
            ax.axvline(x, color="w", ls="--", lw=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) MPI solution at t_f
    ax = axes[0, 0]
    im = ax.imshow(mpi.f().T, extent=extent, origin="lower", aspect=0.5)
    mark_interfaces(ax)
    ax.set_title(f"(a) MPI, {size} ranks, {mpi.n_domains} domains, $t=${T_F}")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\mu$")
    fig.colorbar(im, ax=ax)

    # (b) pointwise difference to serial at t_f
    ax = axes[0, 1]
    diff = np.abs(mpi.f() - ser.f())
    identical = not diff.any()
    im = ax.imshow(np.maximum(diff, FLOOR).T, extent=extent, origin="lower",
                   aspect=0.5, norm=LogNorm(vmin=FLOOR, vmax=max(diff.max(), 1e-15)))
    mark_interfaces(ax)
    ax.set_title(r"(b) $|f_{MPI} - f_{serial}|$ at $t_f$"
                 + (" (bit-identical)" if identical else ""))
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\mu$")
    fig.colorbar(im, ax=ax)

    # (c) relative error over time
    ax = axes[1, 0]
    t = np.array(ser.snap_t)
    ax.semilogy(t, np.maximum(err_mpi["rel_l2"], FLOOR), label="MPI vs serial")
    ax.semilogy(t, np.maximum(err_ctrl["rel_l2"], FLOOR),
                label="MPI without exchange vs serial (control)")
    ax.semilogy(t, np.maximum(err_dd["rel_l2"], FLOOR),
                label="DD vs one domain (DD error)")
    ax.axhline(FLOOR, color="k", lw=0.5, ls=":")
    ax.text(t[-1], FLOOR, "0 (bit-identical)", ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title("(c) Relative error over time")
    ax.legend(fontsize=8)

    # (d) rank history, serial wide and light, MPI dashed on top.
    # The problem is symmetric, so mirrored domains can lie on top of each other.
    ax = axes[1, 1]
    for i, (r_ser, r_mpi) in enumerate(zip(ser.rank_hist, mpi.rank_hist)):
        ax.plot(ser.time, r_ser, color=f"C{i}", lw=4, alpha=0.4,
                label=f"serial, domain {i}")
        ax.plot(mpi.time, r_mpi, "--", color=f"C{i}", lw=1.5,
                label=f"MPI, domain {i} (rank {mpi.owner[i]})")
    ax.plot(single.time, single.rank_hist[0], ":", color="grey", label="one domain")
    ax.set_xlabel("$t$")
    ax.set_ylabel("rank $r(t)$")
    ax.set_title("(d) Rank history")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(savepath / f"compare_serial_mpi_{tag}.png", dpi=200)

    # Save all curves, so the plots can be redone without running again
    np.savez(savepath / f"compare_serial_mpi_{tag}.npz",
             t=t, f_ser=ser.f(), f_mpi=mpi.f(),
             rel_mpi=err_mpi["rel_l2"], rel_ctrl=err_ctrl["rel_l2"],
             rel_dd=err_dd["rel_l2"], max_abs_mpi=err_mpi["max_abs"],
             time=np.array(ser.time), ranks_ser=np.array(ser.rank_hist),
             ranks_mpi=np.array(mpi.rank_hist), owner=np.array(mpi.owner),
             time_single=np.array(single.time),
             ranks_single=np.array(single.rank_hist[0]))


### Main
if __name__ == "__main__":
    n_domains = int(sys.argv[1]) if len(sys.argv) > 1 else size

    mpi = run_mpi(n_domains, use_neighbours=True)
    ctrl = run_mpi(n_domains, use_neighbours=False)

    if rank == 0:
        ser = run_serial(n_domains)
        single = run_serial(1)

        err_mpi = report("MPI vs serial", mpi, ser)
        err_ctrl = report("MPI without exchange vs serial (control)", ctrl, ser)
        err_dd = report("DD vs one domain", ser, single)

        savepath = Path(__file__).resolve().parents[3] / "plots"
        savepath.mkdir(parents=True, exist_ok=True)
        plot_comparison(ser, mpi, single, err_mpi, err_ctrl, err_dd, savepath,
                        tag=f"{size}ranks_{n_domains}domains")

        # Fails loudly if a change breaks the MPI version
        assert mpi.rank_hist == ser.rank_hist, "MPI changes the rank history!"
        assert np.nanmax(err_mpi["rel_l2"]) < 1e-12, "MPI changes the result!"
        print("\nTest passed")
