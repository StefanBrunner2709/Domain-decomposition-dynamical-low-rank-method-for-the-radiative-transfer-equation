import os

# BLAS and LAPACK to 1 for each MPI RANK
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path  

import matplotlib.pyplot as plt  
import numpy as np  
from mpi4py import MPI  
from tqdm import tqdm  

from DLR_rt.src.grid import Grid_1x1d  
from DLR_rt.src.initial_condition import setInitialCondition_1x1d_lr  
from DLR_rt.src.integrators import PSI_lie  
from DLR_rt.src.lr import (  
    add_basis_functions,
    computeF_b_edges,
    drop_basis_functions,
    first_row,
    last_row,
)

# MPI Setup
# COMM_WORLD is the default communicator that includes all processes
# rank is the unique identifier for each process (0, 1, 2, ..., size-1)
# size is the total number of processes in the communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Neighbours, PROC_NULL at the physical boundaries (left and right)
left_nb = rank - 1 if rank > 0 else MPI.PROC_NULL
right_nb = rank + 1 if rank < size - 1 else MPI.PROC_NULL


def integrate(lr, grid, t_f, dt, tol_sing_val=1e-6, drop_tol=1e-6,
              use_neighbours=True):
    t = 0
    time = [t]
    rank_hist = [grid.r]
    timings = {"exchange": 0.0, "F_b": 0.0, "add_basis": 0.0,
               "PSI": 0.0, "drop_basis": 0.0}

    # Receive buffers for the neighbour rows
    from_left = np.empty(grid.Nmu)
    from_right = np.empty(grid.Nmu)

    # Use the received rows only at an interface, None gives tanh(t).
    # use_neighbours=False is the negative control: domains ignore each other.
    has_left = use_neighbours and left_nb != MPI.PROC_NULL
    has_right = use_neighbours and right_nb != MPI.PROC_NULL

    with tqdm(total=t_f / dt, desc="Running Simulation", disable=rank != 0) as pbar:
        while t < t_f:
            pbar.update(1)

            if t + dt > t_f:
                dt = t_f - t

            # Exchange only one row of length Nmu per neighbour
            t0 = MPI.Wtime()
            # Sendrecv sends and receives data in a single call
            comm.Sendrecv(last_row(lr), dest=right_nb,
                          recvbuf=from_left, source=left_nb)
            comm.Sendrecv(first_row(lr), dest=left_nb,
                          recvbuf=from_right, source=right_nb)
            timings["exchange"] += MPI.Wtime() - t0

            # Compute F_b, None at the physical boundary --> tanh(t)
            t0 = MPI.Wtime()
            F_b = computeF_b_edges(
                t, lr, grid,
                inflow_left=from_left if has_left else None,
                inflow_right=from_right if has_right else None,
            )
            timings["F_b"] += MPI.Wtime() - t0

            # Local update same as in serial version
            t0 = MPI.Wtime()
            lr, grid = add_basis_functions(lr, grid, F_b, tol_sing_val)
            timings["add_basis"] += MPI.Wtime() - t0

            t0 = MPI.Wtime()
            lr, grid, _, _ = PSI_lie(lr, grid, dt, F_b, option_scheme="upwind")
            timings["PSI"] += MPI.Wtime() - t0

            t0 = MPI.Wtime()
            lr, grid = drop_basis_functions(lr, grid, drop_tol)
            timings["drop_basis"] += MPI.Wtime() - t0

            t += dt
            time.append(t)
            rank_hist.append(grid.r)

    return lr, time, rank_hist, timings


if __name__ == "__main__":
    assert size == 2, "This version is written for exactly 2 ranks"

    seed = 0
    Nx = 256
    Nmu = 128
    dt = 1e-3
    r = 5
    t_f = 1.0
    tol = 1e-5
    # Run serial dlr_dd on rank 0 afterwards, compare result and runtime
    run_serial_test = True
    savepath = Path(__file__).resolve().parents[3] / "plots"
    savepath.mkdir(parents=True, exist_ok=True)

    # Every rank builds both initial conditions in the same order as serial.
    np.random.seed(seed)
    grids = Grid_1x1d(Nx, Nmu, r).split()
    lr0s = [setInitialCondition_1x1d_lr(g) for g in grids]
    grid, lr0 = grids[rank], lr0s[rank]
    extent = [grids[0].X[0], grids[1].X[-1], grids[0].MU[0], grids[0].MU[-1]]

    # start the timings on all ranks together
    comm.Barrier()
    t0 = MPI.Wtime()
    lr, time, rank_hist, timings = integrate(
        lr0, grid, t_f, dt, tol_sing_val=tol, drop_tol=tol,
        use_neighbours=True,
    )
    runtime = MPI.Wtime() - t0

    # Collect everything on rank 0 for plotting and saving
    f_local = lr.U @ lr.S @ lr.V.T
    all_f = comm.gather(f_local, root=0)
    all_ranks = comm.gather(rank_hist, root=0)
    all_timings = comm.gather(timings, root=0)
    all_runtimes = comm.gather(runtime, root=0)

    if rank == 0:
        f = np.concatenate(all_f, axis=0)
        # The parallel run takes as long as its slowest rank
        runtime_mpi = max(all_runtimes)

        print(f"\nMPI runtime (slowest of {size} ranks): {runtime_mpi:.2f} s")
        for i, tim in enumerate(all_timings):
            print(f"Rank {i}: " + ", ".join(f"{k} {v:.2f} s" for k, v in tim.items()))

        if run_serial_test:
            from DLR_rt.examples.d1x1 import dlr_dd

            # Same initial condition and timing window as the MPI run
            np.random.seed(seed)
            g_l, g_r = Grid_1x1d(Nx, Nmu, r).split()
            lr_l = setInitialCondition_1x1d_lr(g_l)
            lr_r = setInitialCondition_1x1d_lr(g_r)

            t0 = MPI.Wtime()
            lr_l, lr_r, _, rank_l, rank_r, timings_ser, _ = dlr_dd.integrate(
                lr_l, lr_r, g_l, g_r, t_f, dt, tol_sing_val=tol, drop_tol=tol
            )
            runtime_ser = MPI.Wtime() - t0

            f_ref = np.concatenate((lr_l.U @ lr_l.S @ lr_l.V.T,
                                    lr_r.U @ lr_r.S @ lr_r.V.T), axis=0)
            rel_diff = np.linalg.norm(f - f_ref) / np.linalg.norm(f_ref)
            same_rank = all_ranks == [rank_l, rank_r]

            print("\n### Comparison with serial dlr_dd")
            print(f"Bit-identical: {np.array_equal(f, f_ref)}")
            print(f"Relative difference: {rel_diff:.2e}")
            print(f"Same rank history: {same_rank}")

            speedup = runtime_ser / runtime_mpi
            print(f"\nSerial runtime: {runtime_ser:.2f} s")
            print(f"MPI runtime:    {runtime_mpi:.2f} s")
            print(f"Speedup: {speedup:.2f}, parallel efficiency: "
                  f"{speedup / size * 100:.0f} %")

            # Per phase: serial is the sum over both domains,
            # MPI is the slowest rank in that phase
            print(f"\n{'phase':<12}{'serial [s]':>12}{'MPI max [s]':>14}")
            for key in ["F_b", "add_basis", "PSI", "drop_basis"]:
                t_ser = sum(v for k, v in timings_ser.items() if k.startswith(key))
                t_mpi = max(tim[key] for tim in all_timings)
                print(f"{key:<12}{t_ser:>12.2f}{t_mpi:>14.2f}")
            print(f"{'exchange':<12}{'-':>12}"
                  f"{max(tim['exchange'] for tim in all_timings):>14.2f}")    

        ### Plot, only rank 0
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(f.T, extent=extent, origin="lower", aspect=0.5)
        ax.set_xlabel("$x$")
        ax.set_ylabel(r"$\mu$")
        ax.set_title(f"MPI, {size} ranks, $t=${t_f}")
        fig.colorbar(im, ax=ax)
        plt.savefig(savepath / f"dd_mpi_t{t_f}.pdf")
