import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from DLR_rt.src.grid import Grid_1x1d
from DLR_rt.src.initial_condition import setInitialCondition_1x1d_lr
from DLR_rt.src.integrators import PSI_lie, PSI_strang
from DLR_rt.src.lr import LR, add_basis_functions, computeF_b, drop_basis_functions


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
):
    lr_left = lr0_left
    lr_right = lr0_right
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

            ### Update left side

            ### Add basis for adaptive rank strategy:
            lr_left, grid_left = add_basis_functions(
                lr_left, grid_left, F_b_left, tol_sing_val
            )

            ### Run PSI
            if option == "lie":
                lr_left, grid_left = PSI_lie(lr_left, grid_left, dt, F_b_left)

            if option == "strang":
                lr_left, grid_left = PSI_strang(lr_left, grid_left, dt, t, F_b_left)

            ### Drop basis for adaptive rank strategy:
            lr_left, grid_left = drop_basis_functions(lr_left, grid_left, drop_tol)

            ### Update right side

            ### Add basis for adaptive rank strategy:
            lr_right, grid_right = add_basis_functions(
                lr_right, grid_right, F_b_right, tol_sing_val
            )

            ### Run PSI
            if option == "lie":
                lr_right, grid_right = PSI_lie(lr_right, grid_right, dt, F_b_right)

            if option == "strang":
                lr_right, grid_right = PSI_strang(
                    lr_right, grid_right, dt, t, F_b_right
                )

            ### Drop basis for adaptive rank strategy:
            lr_right, grid_right = drop_basis_functions(lr_right, grid_right, drop_tol)

            # Update time
            t += dt
            time.append(t)

            # adapt_rank.append(grid.r)

    return lr_left, lr_right, time


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
lr0_left = setInitialCondition_1x1d_lr(grid_left)
lr0_right = setInitialCondition_1x1d_lr(grid_right)
extent = [grid_left.X[0], grid_right.X[-1], grid_left.MU[0], grid_left.MU[-1]]

lr_left, lr_right, time = integrate(
    lr0_left,
    lr0_right,
    grid_left,
    grid_right,
    t_f,
    dt,
    option=method,
    tol_sing_val=1e-5,
    drop_tol=1e-5,
)
f_left = lr_left.U @ lr_left.S @ lr_left.V.T
f_right = lr_right.U @ lr_right.S @ lr_right.V.T
# Concatenate left and right domain
f = np.concatenate((f_left, f_right), axis=0)

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
    + "dd_distr_funct_initalltanh_fixedcol_t"
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
