import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from DLR_rt.src.grid import Grid_1x1d
from DLR_rt.src.initial_condition import setInitialCondition_1x1d_lr
from DLR_rt.src.integrators import PSI_lie, PSI_strang
from DLR_rt.src.lr import LR, add_basis_functions, computeF_b, drop_basis_functions


def integrate(
    lr0_1: LR,
    lr0_2: LR,
    lr0_3: LR,
    lr0_4: LR,
    grid_1,
    grid_2,
    grid_3,
    grid_4,
    t_f: float,
    dt: float,
    option: str = "lie",
    tol_sing_val: float = 1e-6,
    drop_tol: float = 1e-6,
):
    lr_1 = lr0_1
    lr_2 = lr0_2
    lr_3 = lr0_3
    lr_4 = lr0_4
    t = 0
    time = []
    time.append(t)
    # adapt_rank = []
    # adapt_rank.append(grid.r)

    with tqdm(total=t_f / dt, desc="Running Simulation") as pbar:
        while t < t_f:
            pbar.update(1)

            if t + dt > t_f:
                dt = t_f - t

            ### Compute F_b
            F_b_1 = computeF_b(
                t,
                lr_1.U @ lr_1.S @ lr_1.V.T,
                grid_1,
                f_right=lr_2.U @ lr_2.S @ lr_2.V.T,
            )
            F_b_2 = computeF_b(
                t,
                lr_2.U @ lr_2.S @ lr_2.V.T,
                grid_2,
                f_left=lr_1.U @ lr_1.S @ lr_1.V.T,
                f_right=lr_3.U @ lr_3.S @ lr_3.V.T,
            )
            F_b_3 = computeF_b(
                t,
                lr_3.U @ lr_3.S @ lr_3.V.T,
                grid_3,
                f_left=lr_2.U @ lr_2.S @ lr_2.V.T,
                f_right=lr_4.U @ lr_4.S @ lr_4.V.T,
            )
            F_b_4 = computeF_b(
                t, lr_4.U @ lr_4.S @ lr_4.V.T, grid_4, f_left=lr_3.U @ lr_3.S @ lr_3.V.T
            )

            ### Update subdomain 1

            ### Add basis for adaptive rank strategy:
            lr_1, grid_1 = add_basis_functions(lr_1, grid_1, F_b_1, tol_sing_val)

            ### Run PSI
            if option == "lie":
                lr_1, grid_1 = PSI_lie(lr_1, grid_1, dt, F_b_1)

            if option == "strang":
                lr_1, grid_1 = PSI_strang(lr_1, grid_1, dt, t, F_b_1)

            ### Drop basis for adaptive rank strategy:
            lr_1, grid_1 = drop_basis_functions(lr_1, grid_1, drop_tol)

            ### Update subdomain 2

            ### Add basis for adaptive rank strategy:
            lr_2, grid_2 = add_basis_functions(lr_2, grid_2, F_b_2, tol_sing_val)

            ### Run PSI
            if option == "lie":
                lr_2, grid_2 = PSI_lie(lr_2, grid_2, dt, F_b_2)

            if option == "strang":
                lr_2, grid_2 = PSI_strang(lr_2, grid_2, dt, t, F_b_2)

            ### Drop basis for adaptive rank strategy:
            lr_2, grid_2 = drop_basis_functions(lr_2, grid_2, drop_tol)

            ### Update subdomain 3

            ### Add basis for adaptive rank strategy:
            lr_3, grid_3 = add_basis_functions(lr_3, grid_3, F_b_3, tol_sing_val)

            ### Run PSI
            if option == "lie":
                lr_3, grid_3 = PSI_lie(lr_3, grid_3, dt, F_b_3)

            if option == "strang":
                lr_3, grid_3 = PSI_strang(lr_3, grid_3, dt, t, F_b_3)

            ### Drop basis for adaptive rank strategy:
            lr_3, grid_3 = drop_basis_functions(lr_3, grid_3, drop_tol)

            ### Update subdomain 4

            ### Add basis for adaptive rank strategy:
            lr_4, grid_4 = add_basis_functions(lr_4, grid_4, F_b_4, tol_sing_val)

            ### Run PSI
            if option == "lie":
                lr_4, grid_4 = PSI_lie(lr_4, grid_4, dt, F_b_4)

            if option == "strang":
                lr_4, grid_4 = PSI_strang(lr_4, grid_4, dt, t, F_b_4)

            ### Drop basis for adaptive rank strategy:
            lr_4, grid_4 = drop_basis_functions(lr_4, grid_4, drop_tol)

            # Update time
            t += dt
            time.append(t)

            # adapt_rank.append(grid.r)

    return lr_1, lr_2, lr_3, lr_4, time


### Just one plot for certain rank and certain time

Nx = 64
Nmu = 64
dt = 1e-4
r = 5
t_f = 0.5
fs = 30
method = "lie"
savepath = "plots/"

fig, axes = plt.subplots(1, 1, figsize=(10, 8))

grid = Grid_1x1d(Nx, Nmu, r)
grid_left, grid_right = grid.split()
grid_1, grid_2 = grid_left.split()
grid_3, grid_4 = grid_right.split()
lr0_1 = setInitialCondition_1x1d_lr(grid_1)
lr0_2 = setInitialCondition_1x1d_lr(grid_2)
lr0_3 = setInitialCondition_1x1d_lr(grid_3)
lr0_4 = setInitialCondition_1x1d_lr(grid_4)
extent = [grid_left.X[0], grid_right.X[-1], grid_left.MU[0], grid_left.MU[-1]]

lr_1, lr_2, lr_3, lr_4, time = integrate(
    lr0_1,
    lr0_2,
    lr0_3,
    lr0_4,
    grid_1,
    grid_2,
    grid_3,
    grid_4,
    t_f,
    dt,
    option=method,
    tol_sing_val=1e-5,
    drop_tol=1e-5,
)
f_1 = lr_1.U @ lr_1.S @ lr_1.V.T
f_2 = lr_2.U @ lr_2.S @ lr_2.V.T
f_3 = lr_3.U @ lr_3.S @ lr_3.V.T
f_4 = lr_4.U @ lr_4.S @ lr_4.V.T
# Concatenate left and right domain
f = np.concatenate((f_1, f_2, f_3, f_4), axis=0)

im = axes.imshow(f.T, extent=extent, origin="lower", aspect=0.5, vmin=0.0, vmax=1.0)
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel(r"$\mu$", fontsize=fs, labelpad=-5)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([-1, 0, 1])
axes.tick_params(axis="both", labelsize=fs, pad=20)
axes.set_title("$t=$" + str(t_f), fontsize=fs)

cbar_fixed = fig.colorbar(im, ax=axes)
cbar_fixed.set_ticks([0, 0.5, 1])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(
    savepath
    + "dd_4domains_distr_funct_initalltanh_fixedcol_t"
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
