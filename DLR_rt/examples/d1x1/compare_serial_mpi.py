"""
Compare the MPI version (dlr_dd_mpi) against the serial version (dlr_dd).

Two controls show the scale of a real error:
- MPI without neighbour exchange (negative control, domains ignore each other)
- serial dlr_dd against one domain without decomposition (dlr_inflow)
  this is the actual error of the domain decomposition
"""
import os

# BLAS and LAPACK to 1 for each MPI rank
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpi4py import MPI

from DLR_rt.examples.d1x1 import dlr_dd, dlr_dd_mpi, dlr_inflow
from DLR_rt.src.grid import Grid_1x1d
from DLR_rt.src.initial_condition import setInitialCondition_1x1d_lr

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


### Setup (same for every run)
def setup():
    """Split grid and initial condition, built in the same order as dlr_dd."""
    np.random.seed(SEED)
    grids = Grid_1x1d(NX, NMU, R0).split()
    lr0s = [setInitialCondition_1x1d_lr(g) for g in grids]
    return grids, lr0s


def domain_rng(i):
    """Own random generator for subdomain i, same seed in serial and MPI."""
    return np.random.default_rng([SEED, i])


### Runs, each returns dict(t, f [n_snap, Nx, Nmu], time, ranks [n_dom][n_t])
def run_mpi(use_neighbours=True):
    """All ranks together, result only on rank 0."""
    grids, lr0s = setup()
    snaps = []
    _, time, rank_hist, _ = dlr_dd_mpi.integrate(
        lr0s[rank], grids[rank], T_F, DT, tol_sing_val=TOL, drop_tol=TOL,
        use_neighbours=use_neighbours, rng=domain_rng(rank),
        snapshots=snaps, snap_every=SNAP_EVERY,
    )
    all_snaps = comm.gather(snaps, root=0)
    all_ranks = comm.gather(rank_hist, root=0)
    if rank != 0:
        return None

    # Glue the subdomains together for every snapshot
    f = np.stack([np.concatenate([s[i][1] for s in all_snaps], axis=0)
                  for i in range(len(snaps))])
    return dict(t=np.array([s[0] for s in snaps]), f=f,
                time=np.array(time), ranks=all_ranks)


def run_serial():
    """Serial dlr_dd with the same seeds, only on rank 0."""
    grids, lr0s = setup()
    snaps = []
    _, _, time, rank_l, rank_r, _, _ = dlr_dd.integrate(
        lr0s[0], lr0s[1], grids[0], grids[1], T_F, DT,
        tol_sing_val=TOL, drop_tol=TOL,
        rng_left=domain_rng(0), rng_right=domain_rng(1),
        snapshots=snaps, snap_every=SNAP_EVERY,
    )
    return dict(t=np.array([s[0] for s in snaps]),
                f=np.stack([s[1] for s in snaps]),
                time=np.array(time), ranks=[rank_l, rank_r])


def run_single_domain():
    """dlr_inflow on the full grid (no DD), only on rank 0."""
    np.random.seed(SEED)
    grid = Grid_1x1d(NX, NMU, R0)
    lr0 = setInitialCondition_1x1d_lr(grid)
    snaps = []
    _, time, adapt_rank, _, _ = dlr_inflow.integrate(
        lr0, grid, T_F, DT, tol_sing_val=TOL, drop_tol=TOL,
        rng=domain_rng(0), snapshots=snaps, snap_every=SNAP_EVERY,
    )
    return dict(t=np.array([s[0] for s in snaps]),
                f=np.stack([s[1] for s in snaps]),
                time=np.array(time), ranks=[adapt_rank])


### Error measures
def errors(a, b):
    """Errors of a against reference b per snapshot."""
    assert np.allclose(a["t"], b["t"]), "Snapshots at different times"
    diff = a["f"] - b["f"]
    # Uniform grid, so the weights dx*dmu cancel in the relative L2 error
    norm_diff = np.linalg.norm(diff, axis=(1, 2))
    norm_ref = np.linalg.norm(b["f"], axis=(1, 2))
    # f(0) = 0, the relative error is not defined there
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(norm_ref > 0, norm_diff / norm_ref, np.nan)
    return dict(max_abs=np.max(np.abs(diff), axis=(1, 2)), rel_l2=rel)


def report(name, a, b):
    """Print the comparison of a against reference b."""
    err = errors(a, b)
    print(f"\n### {name}")
    print(f"Bit-identical:         {np.array_equal(a['f'], b['f'])}")
    print(f"Max abs error at t_f:  {err['max_abs'][-1]:.2e}")
    print(f"Rel. L2 error at t_f:  {err['rel_l2'][-1]:.2e}")
    if len(a["ranks"]) == len(b["ranks"]):
        same_rank = a["ranks"] == b["ranks"]
        print(f"Same rank history:     {same_rank}")
        if not same_rank:
            n = next(i for i in range(len(a["time"]))
                     if any(ra[i] != rb[i] for ra, rb in zip(a["ranks"], b["ranks"])))
            print(f"Rank history differs first at step {n}, t = {a['time'][n]:.3f}")
    return err


### Plots
def plot_comparison(ser, mpi, ctrl, single, err_mpi, err_ctrl, err_dd,
                    grids, savepath):
    extent = [grids[0].X[0], grids[1].X[-1], grids[0].MU[0], grids[0].MU[-1]]
    x_interface = 0.5 * (grids[0].X[-1] + grids[1].X[0])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) MPI solution at t_f
    ax = axes[0, 0]
    im = ax.imshow(mpi["f"][-1].T, extent=extent, origin="lower", aspect=0.5)
    ax.axvline(x_interface, color="w", ls="--", lw=1)
    ax.set_title(f"(a) MPI, {size} ranks, $t=${T_F}")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\mu$")
    fig.colorbar(im, ax=ax)

    # (b) pointwise difference to serial at t_f
    ax = axes[0, 1]
    diff = np.abs(mpi["f"][-1] - ser["f"][-1])
    identical = not diff.any()
    im = ax.imshow(np.maximum(diff, FLOOR).T, extent=extent, origin="lower",
                   aspect=0.5, norm=LogNorm(vmin=FLOOR, vmax=max(diff.max(), 1e-15)))
    ax.axvline(x_interface, color="w", ls="--", lw=1)
    ax.set_title(r"(b) $|f_{MPI} - f_{serial}|$ at $t_f$"
                 + (" (bit-identical)" if identical else ""))
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\mu$")
    fig.colorbar(im, ax=ax)

    # (c) relative error over time
    ax = axes[1, 0]
    t = ser["t"]
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
    # The problem is symmetric, so both domains can lie on top of each other.
    ax = axes[1, 1]
    for i, (r_ser, r_mpi) in enumerate(zip(ser["ranks"], mpi["ranks"])):
        ax.plot(ser["time"], r_ser, color=f"C{i}", lw=4, alpha=0.4,
                label=f"serial, domain {i}")
        ax.plot(mpi["time"], r_mpi, "--", color=f"C{i}", lw=1.5,
                label=f"MPI, rank {i}")
    ax.plot(single["time"], single["ranks"][0], ":", color="grey",
            label="one domain")
    ax.set_xlabel("$t$")
    ax.set_ylabel("rank $r(t)$")
    ax.set_title("(d) Rank history")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(savepath / f"compare_serial_mpi_{size}ranks.pdf")

    # Save all curves, so the plots can be redone without running again
    np.savez(savepath / f"compare_serial_mpi_{size}ranks.npz",
             t=t, f_ser=ser["f"][-1], f_mpi=mpi["f"][-1],
             rel_mpi=err_mpi["rel_l2"], rel_ctrl=err_ctrl["rel_l2"],
             rel_dd=err_dd["rel_l2"], max_abs_mpi=err_mpi["max_abs"],
             time=ser["time"], ranks_ser=np.array(ser["ranks"]),
             ranks_mpi=np.array(mpi["ranks"]),
             time_single=single["time"], ranks_single=np.array(single["ranks"][0]))


### Main
if __name__ == "__main__":
    assert size == 2, "Serial dlr_dd has 2 domains, so run with exactly 2 ranks"

    mpi = run_mpi(use_neighbours=True)
    ctrl = run_mpi(use_neighbours=False)

    if rank == 0:
        ser = run_serial()
        single = run_single_domain()

        err_mpi = report("MPI vs serial", mpi, ser)
        err_ctrl = report("MPI without exchange vs serial (control)", ctrl, ser)
        err_dd = report("DD vs one domain", ser, single)

        savepath = Path(__file__).resolve().parents[3] / "plots"
        savepath.mkdir(parents=True, exist_ok=True)
        grids, _ = setup()
        plot_comparison(ser, mpi, ctrl, single, err_mpi, err_ctrl, err_dd,
                        grids, savepath)

        # Fails loudly if a change breaks the MPI version
        assert mpi["ranks"] == ser["ranks"], "MPI changes the rank history!"
        assert np.nanmax(err_mpi["rel_l2"]) < 1e-12, "MPI changes the result!"
        print("\nTest passed")
