import os 

# BLAS and LAPACK to 1 for each MPI rank
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt  
import numpy as np  
from mpi4py import MPI  
from pathlib import Path  

from DLR_rt.examples.d1x1 import dlr_dd, dlr_dd_mpi, dlr_inflow  
from DLR_rt.src.grid import Grid_1x1d  
from DLR_rt.src.initial_condition import setInitialCondition_1x1d_lr  

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

NXS = [128, 256, 512, 1024, 2048] # Set the number of grid points in x-direction
NMU = 128
R0 = 5
TOL = 1e-5
T_F = 0.05 # Shorter runs
CFL = 0.25 
N_REPEAT = 3 # take fastest out of three
SEED = 0

def setup(Nx):
    """  
    Same initial conditions
    """
    np.random.seed(SEED)
    grid = Grid_1x1d(Nx,NMU,R0)
    dt = CFL * grid.dx 
    grids = grid.split()
    lr0s = [setInitialCondition_1x1d_lr(g) for g in grids]
    return dt, grids, lr0s


def time_mpi(Nx):
    """Time per step of the MPI version, all ranks call this together."""
    best = np.inf
    for _ in range(N_REPEAT):
        dt, grids, lr0s = setup(Nx)
        comm.Barrier()  # all ranks start together
        t0 = MPI.Wtime()
        _, time, _, _ = dlr_dd_mpi.integrate(
            lr0s[rank], grids[rank], T_F, dt, tol_sing_val=TOL, drop_tol=TOL
        )
        local = MPI.Wtime() - t0
        # The parallel run takes as long as the slowest rank
        runtime = comm.allreduce(local, op=MPI.MAX)
        best = min(best, runtime / (len(time) - 1))
    return best

def time_serial_dd(Nx):
    """Time per step of serial dlr_dd, only on rank 0."""
    best = np.inf
    for _ in range(N_REPEAT):
        dt, grids, lr0s = setup(Nx)
        t0 = MPI.Wtime()
        _, _, time, _, _, _, _ = dlr_dd.integrate(
            lr0s[0], lr0s[1], grids[0], grids[1], T_F, dt,
            tol_sing_val=TOL, drop_tol=TOL,
        )
        best = min(best, (MPI.Wtime() - t0) / (len(time) - 1))
    return best


def time_serial_inflow(Nx):
    """Time per step of serial dlr_inflow (one domain), only on rank 0."""
    best = np.inf
    for _ in range(N_REPEAT):
        np.random.seed(SEED)
        grid = Grid_1x1d(Nx, NMU, R0)
        lr0 = setInitialCondition_1x1d_lr(grid)
        dt = CFL * grid.dx
        t0 = MPI.Wtime()
        _, time, _, _, _ = dlr_inflow.integrate(
            lr0, grid, T_F, dt, tol_sing_val=TOL, drop_tol=TOL
        )
        best = min(best, (MPI.Wtime() - t0) / (len(time) - 1))
    return best


if __name__ == "__main__":
    t_mpi, t_dd, t_inflow = [], [], []

    for Nx in NXS:
        t_mpi.append(time_mpi(Nx))

        # Serial references only on rank 0, the other ranks wait at the Barrier
        if rank == 0:
            t_dd.append(time_serial_dd(Nx))
            t_inflow.append(time_serial_inflow(Nx))
            print(f"Nx={Nx}: inflow {t_inflow[-1] * 1e3:.2f} ms, "
                  f"dd {t_dd[-1] * 1e3:.2f} ms, MPI {t_mpi[-1] * 1e3:.2f} ms")
        comm.Barrier()

    if rank == 0:
        Nx = np.array(NXS)
        t_mpi, t_dd, t_inflow = map(np.array, (t_mpi, t_dd, t_inflow))

        savepath = Path(__file__).resolve().parents[3] / "plots"
        savepath.mkdir(parents=True, exist_ok=True)
        np.savez(savepath / f"mpi_scaling_{size}ranks.npz",
                 Nx=Nx, t_mpi=t_mpi, t_dd=t_dd, t_inflow=t_inflow)

        fig, ax1 = plt.subplots(figsize=(7, 5))

        # Time per step over the discretisation
        ax1.loglog(Nx, t_inflow * 1e3, "o-", label="serial, 1 domain")
        ax1.loglog(Nx, t_dd * 1e3, "s-", label="serial, 2 domains")
        ax1.loglog(Nx, t_mpi * 1e3, "^-", label=f"MPI, {size} ranks")
        ax1.loglog(Nx, t_inflow[0] * 1e3 * Nx / Nx[0], "k--", lw=1,
                   label=r"$\mathcal{O}(N_x)$")
        ax1.set_xlabel("$N_x$")
        ax1.set_ylabel("time per step [ms]")
        ax1.legend()

        plt.tight_layout()
        plt.savefig(savepath / f"mpi_scaling_{size}ranks.pdf")