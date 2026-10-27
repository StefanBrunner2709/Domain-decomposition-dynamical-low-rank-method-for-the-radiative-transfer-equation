# Set BLAS and LAPACK threads to 1 because np sometimes uses more
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import tracemalloc
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt  
import numpy as np  
from tqdm import tqdm  

from DLR_rt.src.grid import Grid_1x1d  # noqa: E402
from DLR_rt.src.initial_condition import setInitialCondition_1x1d_lr  # noqa: E402
from DLR_rt.src.integrators import PSI_lie, PSI_strang  # noqa: E402
from DLR_rt.src.lr import (  # noqa: E402
    LR,
    add_basis_functions,
    computeF_b,
    computeF_b_edges,
    drop_basis_functions,
    first_row,
    last_row,
)
from DLR_rt.src.util import mem_phase_end, mem_phase_start  # noqa: E402


def integrate(
    lr0_left: LR,
    lr0_right: LR,
    grid_left,
    grid_right,
    t_f: float,
    dt: float,
    option: str = "lie",
    tol_sing_val: float = 1e-6,
    drop_tol: float = 1e-6,
    F_b_method: str = "edges",
    rng_left=None,
    rng_right=None,
    snapshots=None,
    snap_every: int = 10,
):
    # One np.random.Generator per subdomain for add_basis_functions, so that
    # the random numbers do not depend on the order in which domains are updated
    # (needed to compare with the MPI version). None uses the global np.random.
    # snapshots: if a list is given, (t, f on the whole domain) is appended
    # every snap_every steps
    lr_left = lr0_left
    lr_right = lr0_right
    t = 0
    step = 0
    if snapshots is not None:
        snapshots.append((t, np.concatenate((lr_left.U @ lr_left.S @ lr_left.V.T,
                                             lr_right.U @ lr_right.S @ lr_right.V.T))))
    time = []
    time.append(t)
    # Rank history per subdomain, one entry per time step
    rank_left = [grid_left.r]
    rank_right = [grid_right.r]

    # Timings, F_b covers both domains since it becomes the exchange in MPI
    timings = {
        "F_b": 0.0,
        "add_basis_left": 0.0,
        "PSI_left": 0.0,
        "drop_basis_left": 0.0,
        "add_basis_right": 0.0,
        "PSI_right": 0.0,
        "drop_basis_right": 0.0,
    }
    # Peak additional memory per phase in bytes
    mem_peak = dict.fromkeys(timings, 0)

    with tqdm(total=t_f / dt, desc="Running Simulation") as pbar:
        while t < t_f:
            pbar.update(1)

            if t + dt > t_f:
                dt = t_f - t

            ### Compute F_b
            t0 = perf_counter()
            m0 = mem_phase_start()
            if F_b_method == "edges":
                # Only two rows need to be interchanged between the domains,
                # TODO this will be the later MPI communication 
                F_b_left = computeF_b_edges(
                    t, lr_left, grid_left, inflow_right=first_row(lr_right)
                )
                F_b_right = computeF_b_edges(
                    t, lr_right, grid_right, inflow_left=last_row(lr_left)
                )
            elif F_b_method == "full":
                F_b_left = computeF_b(
                    t,
                    lr_left.U @ lr_left.S @ lr_left.V.T,
                    grid_left,
                    f_right=lr_right.U @ lr_right.S @ lr_right.V.T,
                )
                F_b_right = computeF_b(
                    t,
                    lr_right.U @ lr_right.S @ lr_right.V.T,
                    grid_right,
                    f_left=lr_left.U @ lr_left.S @ lr_left.V.T,
                )
            mem_phase_end(mem_peak, "F_b", m0)
            timings["F_b"] += perf_counter() - t0

            ### Update left side

            ### Add basis for adaptive rank strategy:
            t0 = perf_counter()
            m0 = mem_phase_start()
            lr_left, grid_left = add_basis_functions(
                lr_left, grid_left, F_b_left, tol_sing_val, rng=rng_left
            )
            mem_phase_end(mem_peak, "add_basis_left", m0)
            timings["add_basis_left"] += perf_counter() - t0

            ### Run PSI
            t0 = perf_counter()
            m0 = mem_phase_start()
            if option == "lie":
                lr_left, grid_left, _, _ = PSI_lie(lr_left, grid_left, dt, F_b_left,
                                                   option_scheme="upwind")

            if option == "strang":
                lr_left, grid_left = PSI_strang(lr_left, grid_left, dt, t, F_b_left)
            mem_phase_end(mem_peak, "PSI_left", m0)
            timings["PSI_left"] += perf_counter() - t0

            ### Drop basis for adaptive rank strategy:
            t0 = perf_counter()
            m0 = mem_phase_start()
            lr_left, grid_left = drop_basis_functions(lr_left, grid_left, drop_tol)
            mem_phase_end(mem_peak, "drop_basis_left", m0)
            timings["drop_basis_left"] += perf_counter() - t0

            ### Update right side

            ### Add basis for adaptive rank strategy:
            t0 = perf_counter()
            m0 = mem_phase_start()
            lr_right, grid_right = add_basis_functions(
                lr_right, grid_right, F_b_right, tol_sing_val, rng=rng_right
            )
            mem_phase_end(mem_peak, "add_basis_right", m0)
            timings["add_basis_right"] += perf_counter() - t0

            ### Run PSI
            t0 = perf_counter()
            m0 = mem_phase_start()
            if option == "lie":
                lr_right, grid_right, _, _ = PSI_lie(lr_right, grid_right,
                                                      dt, F_b_right,
                                                      option_scheme="upwind")

            if option == "strang":
                lr_right, grid_right = PSI_strang(
                    lr_right, grid_right, dt, t, F_b_right
                )
            mem_phase_end(mem_peak, "PSI_right", m0)
            timings["PSI_right"] += perf_counter() - t0

            ### Drop basis for adaptive rank strategy:
            t0 = perf_counter()
            m0 = mem_phase_start()
            lr_right, grid_right = drop_basis_functions(lr_right, grid_right, drop_tol)
            mem_phase_end(mem_peak, "drop_basis_right", m0)
            timings["drop_basis_right"] += perf_counter() - t0

            # Update time
            t += dt
            step += 1
            time.append(t)

            rank_left.append(grid_left.r)
            rank_right.append(grid_right.r)

            if snapshots is not None and (step % snap_every == 0 or t >= t_f):
                snapshots.append((t, np.concatenate(
                    (lr_left.U @ lr_left.S @ lr_left.V.T,
                     lr_right.U @ lr_right.S @ lr_right.V.T))))

    return lr_left, lr_right, time, rank_left, rank_right, timings, mem_peak


### Just one plot for certain rank and certain time
#  Deterministic runs, remove np.random

if __name__ == "__main__":
    
    seed = 0
    np.random.seed(seed)
    
    Nx = 256
    Nmu = 128
    dt = 1e-3
    r = 5
    t_f = 1.0
    fs = 30
    method = "lie"
    # tracemalloc slows the run down, so never use timings from a memory run
    measure_memory = False
    # Compare computeF_b_edges against the old computeF_b (runs a second simulation)
    run_edge_test = True
    # Repo root is three levels above this file, so the script works from any cwd
    savepath = Path(__file__).resolve().parents[3] / "plots"
    savepath.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    
    grid = Grid_1x1d(Nx, Nmu, r)
    grid_left, grid_right = grid.split()
    lr0_left = setInitialCondition_1x1d_lr(grid_left)
    lr0_right = setInitialCondition_1x1d_lr(grid_right)
    extent = [grid_left.X[0], grid_right.X[-1], grid_left.MU[0], grid_left.MU[-1]]
    
    if measure_memory:
        tracemalloc.start()
    t0 = perf_counter()
    lr_left, lr_right, time, rank_left, rank_right, timings, mem_peak = integrate(
        lr0_left,
        lr0_right,
        grid_left,
        grid_right,
        t_f,
        dt,
        option=method,
        tol_sing_val=1e-5,
        drop_tol=1e-5,
        rng_left=np.random.default_rng([seed, 0]),
        rng_right=np.random.default_rng([seed, 1]),
    )
    runtime = perf_counter() - t0
    print(f"Runtime: {runtime:.2f} seconds")
    
    print("Timings:")
    for key, value in timings.items():
        print(f"{key}: {value:.2f} seconds")
    
    if measure_memory:
        tracemalloc.stop()
        print("Timings are distorted by tracemalloc in this run!")
        print("Peak additional memory per phase:")
        for key, value in mem_peak.items():
            print(f"{key}: {value / 1024:.1f} KB")
    
    f_left = lr_left.U @ lr_left.S @ lr_left.V.T
    
    f_right = lr_right.U @ lr_right.S @ lr_right.V.T
    
    # Concatenate left and right domain
    f = np.concatenate((f_left, f_right), axis=0)
    
    # Compute memory usage of f
    mem_lr = lr_left.U.nbytes + lr_left.S.nbytes + lr_left.V.nbytes + lr_right.U.nbytes + lr_right.S.nbytes + lr_right.V.nbytes 
    mem_f = f.nbytes
    print(f"Memory usage of f: {mem_f / 1024**2:.2f} MB")
    print(f"Memory usage of low-rank representation: {mem_lr / 1024**2:.2f} MB")
    print(f"Memory reduction factor: {mem_f / mem_lr:.2f}")
    
    # save f object
    np.save(savepath / "det_check_dd.npy", f)

    # Rank history per subdomain
    print(f"Rank left:  min {min(rank_left)}, max {max(rank_left)}, "
          f"mean {np.mean(rank_left):.2f}")
    print(f"Rank right: min {min(rank_right)}, max {max(rank_right)}, "
          f"mean {np.mean(rank_right):.2f}")
    np.savez(savepath / "dd_rank_history.npz",
             time=time, rank_left=rank_left, rank_right=rank_right)

    fig_rank, ax_rank = plt.subplots(figsize=(8, 5))
    ax_rank.plot(time, rank_left, label="left domain")
    ax_rank.plot(time, rank_right, label="right domain")
    ax_rank.set_xlabel("$t$", fontsize=fs)
    ax_rank.set_ylabel("rank $r(t)$", fontsize=fs)
    ax_rank.tick_params(axis="both", labelsize=fs)
    ax_rank.margins(x=0)
    ax_rank.legend(fontsize=fs * 0.6)
    fig_rank.tight_layout()
    fig_rank.savefig(savepath / ("dd_rank_history_t" + str(t_f) + ".pdf"))

    if run_edge_test:
        print("\n### Test computeF_b_edges against computeF_b")

        # Test 1: both functions on the same final state with a nonzero f
        t_test = 0.5
        diff_left = np.max(np.abs(
            computeF_b(t_test, f_left, grid_left, f_right=f_right)
            - computeF_b_edges(t_test, lr_left, grid_left,
                               inflow_right=first_row(lr_right))
        ))
        diff_right = np.max(np.abs(
            computeF_b(t_test, f_right, grid_right, f_left=f_left)
            - computeF_b_edges(t_test, lr_right, grid_right,
                               inflow_left=last_row(lr_left))
        ))
        print(f"Single F_b call, max abs difference left: {diff_left:.2e}, "
              f"right: {diff_right:.2e}")
        assert max(diff_left, diff_right) < 1e-14

        # Test 2: whole simulation with the old function, same seed.
        # integrate changes the lr objects and grids in place, so build them again.
        np.random.seed(seed)
        grid_ref_left, grid_ref_right = Grid_1x1d(Nx, Nmu, r).split()
        lr0_ref_left = setInitialCondition_1x1d_lr(grid_ref_left)
        lr0_ref_right = setInitialCondition_1x1d_lr(grid_ref_right)
        lr_ref_left, lr_ref_right, _, rank_ref_left, rank_ref_right, _, _ = integrate(
            lr0_ref_left, lr0_ref_right, grid_ref_left, grid_ref_right, t_f, dt,
            option=method, tol_sing_val=1e-5, drop_tol=1e-5, F_b_method="full",
            rng_left=np.random.default_rng([seed, 0]),
            rng_right=np.random.default_rng([seed, 1]),
        )
        f_ref = np.concatenate((lr_ref_left.U @ lr_ref_left.S @ lr_ref_left.V.T,
                                lr_ref_right.U @ lr_ref_right.S @ lr_ref_right.V.T),
                               axis=0)
        rel_diff = np.linalg.norm(f - f_ref) / np.linalg.norm(f_ref)
        same_rank = rank_left == rank_ref_left and rank_right == rank_ref_right
        print(f"Whole simulation, bit-identical: {np.array_equal(f, f_ref)}")
        print(f"Whole simulation, relative difference: {rel_diff:.2e}")
        print(f"Same rank history (left, right): {same_rank}")
        if not same_rank:
            # First time step where one of the truncation decisions differs
            n = next(i for i in range(len(rank_left))
                     if rank_left[i] != rank_ref_left[i]
                     or rank_right[i] != rank_ref_right[i])
            print(f"Rank history differs first at step {n}, t = {time[n]:.3f}")
        # Like in dlr_inflow: round-off differences in F_b are amplified by the
        # rank truncation, so we expect agreement up to roughly the tolerance.
        assert same_rank, "computeF_b_edges changes the rank history!"
        assert rel_diff < 1e-5, "computeF_b_edges changes the result!"
        print("Test passed")
    
    
    im = axes.imshow(f.T, extent=extent, origin="lower", aspect=0.5)
    axes.set_xlabel("$x$", fontsize=fs)
    axes.set_ylabel(r"$\mu$", fontsize=fs, labelpad=-5)
    axes.set_xticks([0, 0.5, 1])
    axes.set_yticks([-1, 0, 1])
    axes.tick_params(axis="both", labelsize=fs, pad=20)
    axes.set_title("$t=$" + str(t_f), fontsize=fs)
    
    cbar_fixed = fig.colorbar(im, ax=axes)
    # cbar_fixed.set_ticks([0, 0.5, 1])
    cbar_fixed.set_ticks([np.min(f), np.max(f)])
    cbar_fixed.ax.tick_params(labelsize=fs)
    
    plt.tight_layout()
    plt.savefig(
        savepath
        / (
            "dd_distr_funct_initalltanh_fixedcol_t"
            + str(t_f)
            + "_"
            + method
            + "_"
            + str(dt)
            + "_adaptrank_"
            + str(Nx)
            + "x"
            + str(Nmu)
            + ".pdf"
        )
    )
