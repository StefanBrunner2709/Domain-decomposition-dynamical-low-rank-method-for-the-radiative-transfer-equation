import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from DLR_rt.src.grid import Grid_2x1d
from DLR_rt.src.initial_condition import setInitialCondition_2x1d_lr
from DLR_rt.src.integrators import PSI_splitting_lie
from DLR_rt.src.lr import LR, computeF_b_2x1d_X, computeF_b_2x1d_Y
from DLR_rt.src.util import computeD_cendiff_2x1d

# ToDo: Add strang splitting (need to make strang available for domain decomp in Y)


def integrate(
    lr0_left_bottom: LR,
    lr0_left_top: LR,
    lr0_right_bottom: LR,
    lr0_right_top: LR,
    grid_left_bottom: Grid_2x1d,
    grid_left_top: Grid_2x1d,
    grid_right_bottom: Grid_2x1d,
    grid_right_top: Grid_2x1d,
    t_f: float,
    dt: float,
    tol_sing_val: float = 1e-6,
    drop_tol: float = 1e-6,
    method="lie",
    source_left_bottom=None
):
    lr_left_bottom = lr0_left_bottom
    lr_left_top = lr0_left_top
    lr_right_bottom = lr0_right_bottom
    lr_right_top = lr0_right_top
    t = 0
    time = []
    time.append(t)
    rank_left_bottom_adapted = [grid_left_bottom.r]
    rank_left_bottom_dropped = [grid_left_bottom.r]
    rank_left_top_adapted = [grid_left_top.r]
    rank_left_top_dropped = [grid_left_top.r]
    rank_right_bottom_adapted = [grid_right_bottom.r]
    rank_right_bottom_dropped = [grid_right_bottom.r]
    rank_right_top_adapted = [grid_right_top.r]
    rank_right_top_dropped = [grid_right_top.r]

    DX, DY = computeD_cendiff_2x1d(grid_left_bottom, "dd")
    # all grids have same size, thus enough to compute once

    with tqdm(total=t_f / dt, desc="Running Simulation") as pbar:
        while t < t_f:
            pbar.update(1)

            if t + dt > t_f:
                dt = t_f - t

            ### Compute F_b and F_b_top_bottom
            f_left_bottom = lr_left_bottom.U @ lr_left_bottom.S @ lr_left_bottom.V.T
            f_left_top = lr_left_top.U @ lr_left_top.S @ lr_left_top.V.T
            f_right_bottom = lr_right_bottom.U @ lr_right_bottom.S @ lr_right_bottom.V.T
            f_right_top = lr_right_top.U @ lr_right_top.S @ lr_right_top.V.T

            F_b_X_left_bottom = computeF_b_2x1d_X(
                f_left_bottom,
                grid_left_bottom,
                f_right=f_right_bottom,
                f_periodic=f_right_bottom,
            )
            F_b_X_left_top = computeF_b_2x1d_X(
                f_left_top, grid_left_top, f_right=f_right_top, f_periodic=f_right_top
            )
            F_b_X_right_bottom = computeF_b_2x1d_X(
                f_right_bottom,
                grid_right_bottom,
                f_left=f_left_bottom,
                f_periodic=f_left_bottom,
            )
            F_b_X_right_top = computeF_b_2x1d_X(
                f_right_top, grid_right_top, f_left=f_left_top, f_periodic=f_left_top
            )
            F_b_Y_left_bottom = computeF_b_2x1d_Y(
                f_left_bottom, grid_left_bottom, f_top=f_left_top, f_periodic=f_left_top
            )
            F_b_Y_left_top = computeF_b_2x1d_Y(
                f_left_top,
                grid_left_top,
                f_bottom=f_left_bottom,
                f_periodic=f_left_bottom,
            )
            F_b_Y_right_bottom = computeF_b_2x1d_Y(
                f_right_bottom,
                grid_right_bottom,
                f_top=f_right_top,
                f_periodic=f_right_top,
            )
            F_b_Y_right_top = computeF_b_2x1d_Y(
                f_right_top,
                grid_right_top,
                f_bottom=f_right_bottom,
                f_periodic=f_right_bottom,
            )

            ### Update left_bottom side

            ### Run PSI with adaptive rank strategy
            (
                lr_left_bottom,
                grid_left_bottom,
                rank_left_bottom_adapted,
                rank_left_bottom_dropped,
            ) = PSI_splitting_lie(
                lr_left_bottom,
                grid_left_bottom,
                dt,
                F_b_X_left_bottom,
                F_b_Y_left_bottom,
                DX=DX,
                DY=DY,
                tol_sing_val=tol_sing_val,
                drop_tol=drop_tol,
                rank_adapted=rank_left_bottom_adapted,
                rank_dropped=rank_left_bottom_dropped,
                source=source_left_bottom,
            )

            ### Update left_top side

            ### Run PSI with adaptive rank strategy
            (
                lr_left_top,
                grid_left_top,
                rank_left_top_adapted,
                rank_left_top_dropped,
            ) = PSI_splitting_lie(
                lr_left_top,
                grid_left_top,
                dt,
                F_b_X_left_top,
                F_b_Y_left_top,
                DX=DX,
                DY=DY,
                tol_sing_val=tol_sing_val,
                drop_tol=drop_tol,
                rank_adapted=rank_left_top_adapted,
                rank_dropped=rank_left_top_dropped,
            )

            ### Update right_bottom side

            ### Run PSI with adaptive rank strategy
            (
                lr_right_bottom,
                grid_right_bottom,
                rank_right_bottom_adapted,
                rank_right_bottom_dropped,
            ) = PSI_splitting_lie(
                lr_right_bottom,
                grid_right_bottom,
                dt,
                F_b_X_right_bottom,
                F_b_Y_right_bottom,
                DX=DX,
                DY=DY,
                tol_sing_val=tol_sing_val,
                drop_tol=drop_tol,
                rank_adapted=rank_right_bottom_adapted,
                rank_dropped=rank_right_bottom_dropped,
            )

            ### Update right_top side

            ### Run PSI with adaptive rank strategy
            (
                lr_right_top,
                grid_right_top,
                rank_right_top_adapted,
                rank_right_top_dropped,
            ) = PSI_splitting_lie(
                lr_right_top,
                grid_right_top,
                dt,
                F_b_X_right_top,
                F_b_Y_right_top,
                DX=DX,
                DY=DY,
                tol_sing_val=tol_sing_val,
                drop_tol=drop_tol,
                rank_adapted=rank_right_top_adapted,
                rank_dropped=rank_right_top_dropped,
            )

            ### Update time
            t += dt
            time.append(t)

    return (
        lr_left_bottom,
        lr_left_top,
        lr_right_bottom,
        lr_right_top,
        time,
        rank_left_bottom_adapted,
        rank_left_bottom_dropped,
        rank_left_top_adapted,
        rank_left_top_dropped,
        rank_right_bottom_adapted,
        rank_right_bottom_dropped,
        rank_right_top_adapted,
        rank_right_top_dropped,
    )


### Plotting

Nx = 64
Ny = 64
Nphi = 64
dt = 1e-3
r = 5
t_f = 0.5
fs = 16
savepath = "plots/"
method = "lie"


### Initial configuration

grid = Grid_2x1d(Nx, Ny, Nphi, r, _option_dd="dd")
grid_left, grid_right = grid.split_x()

grid_left_bottom, grid_left_top = grid_left.split_y()
grid_right_bottom, grid_right_top = grid_right.split_y()

grid_left_bottom.coeff = [1,0,0]
grid_left_top.coeff = [1,0,0]
grid_right_bottom.coeff = [1,0,0]
grid_right_top.coeff = [1,0,0]

# Set source in left_bottom grid
source_left_bottom = np.ones((grid_left_bottom.Nx, grid_left_bottom.Ny))
# for i in range(grid_left_bottom.Nx):
#     for j in range(grid_left_bottom.Ny):
#         source_left_bottom[i,j] = (
#                         1
#                         / (2 * np.pi)
#                         * np.exp(-((grid.X[i] - 0.25) ** 2) / 0.02)
#                         * np.exp(-((grid.Y[j] - 0.25) ** 2) / 0.02)
#                     )

source_left_bottom = source_left_bottom.flatten()[:, None]

lr0_left_bottom = setInitialCondition_2x1d_lr(grid_left_bottom, option_cond="lattice")
lr0_left_top = setInitialCondition_2x1d_lr(grid_left_top, option_cond="lattice")
lr0_right_bottom = setInitialCondition_2x1d_lr(grid_right_bottom, option_cond="lattice")
lr0_right_top = setInitialCondition_2x1d_lr(grid_right_top, option_cond="lattice")

f0_left_bottom = lr0_left_bottom.U @ lr0_left_bottom.S @ lr0_left_bottom.V.T
f0_left_top = lr0_left_top.U @ lr0_left_top.S @ lr0_left_top.V.T
f0_right_bottom = lr0_right_bottom.U @ lr0_right_bottom.S @ lr0_right_bottom.V.T
f0_right_top = lr0_right_top.U @ lr0_right_top.S @ lr0_right_top.V.T

rho0_left_bottom = (
    f0_left_bottom @ np.ones(grid_left_bottom.Nphi)
) * grid_left_bottom.dphi  # This is now a vector, only depends on x and y
rho0_left_top = (
    f0_left_top @ np.ones(grid_left_top.Nphi)
) * grid_left_top.dphi  # This is now a vector, only depends on x and y
rho0_right_bottom = (
    f0_right_bottom @ np.ones(grid_right_bottom.Nphi)
) * grid_right_bottom.dphi  # This is now a vector, only depends on x and y
rho0_right_top = (
    f0_right_top @ np.ones(grid_right_top.Nphi)
) * grid_right_top.dphi  # This is now a vector, only depends on x and y

rho0_matrix_left_bottom = rho0_left_bottom.reshape(
    (grid_left_bottom.Nx, grid_left_bottom.Ny), order="F"
)
rho0_matrix_left_top = rho0_left_top.reshape(
    (grid_left_top.Nx, grid_left_top.Ny), order="F"
)
rho0_matrix_right_bottom = rho0_right_bottom.reshape(
    (grid_right_bottom.Nx, grid_right_bottom.Ny), order="F"
)
rho0_matrix_right_top = rho0_right_top.reshape(
    (grid_right_top.Nx, grid_right_top.Ny), order="F"
)

rho0_matrix_left = np.concatenate(
    (rho0_matrix_left_bottom, rho0_matrix_left_top), axis=1
)
rho0_matrix_right = np.concatenate(
    (rho0_matrix_right_bottom, rho0_matrix_right_top), axis=1
)

rho0_matrix = np.concatenate((rho0_matrix_left, rho0_matrix_right), axis=0)

extent = [grid.X[0], grid.X[-1], grid.Y[0], grid.Y[-1]]

fig, axes = plt.subplots(1, 1, figsize=(10, 8))

im = axes.imshow(rho0_matrix.T, extent=extent, origin="lower")
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel("$y$", fontsize=fs)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([0, 0.5, 1])
axes.tick_params(axis="both", labelsize=fs, pad=10)

cbar_fixed = fig.colorbar(im, ax=axes)
cbar_fixed.set_ticks([np.min(rho0_matrix), np.max(rho0_matrix)])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "dd_splitting_2x1d_rho_initial.pdf")


### Final configuration

(
    lr_left_bottom,
    lr_left_top,
    lr_right_bottom,
    lr_right_top,
    time,
    rank_left_bottom_adapted,
    rank_left_bottom_dropped,
    rank_left_top_adapted,
    rank_left_top_dropped,
    rank_right_bottom_adapted,
    rank_right_bottom_dropped,
    rank_right_top_adapted,
    rank_right_top_dropped,
) = integrate(
    lr0_left_bottom,
    lr0_left_top,
    lr0_right_bottom,
    lr0_right_top,
    grid_left_bottom,
    grid_left_top,
    grid_right_bottom,
    grid_right_top,
    t_f,
    dt,
    method=method,
    tol_sing_val=1e-3,
    drop_tol=1e-7,
    source_left_bottom=source_left_bottom,
)

f_left_bottom = lr_left_bottom.U @ lr_left_bottom.S @ lr_left_bottom.V.T
f_left_top = lr_left_top.U @ lr_left_top.S @ lr_left_top.V.T
f_right_bottom = lr_right_bottom.U @ lr_right_bottom.S @ lr_right_bottom.V.T
f_right_top = lr_right_top.U @ lr_right_top.S @ lr_right_top.V.T

rho_left_bottom = (
    f_left_bottom @ np.ones(grid_left_bottom.Nphi)
) * grid_left_bottom.dphi  # This is now a vector, only depends on x and y
rho_left_top = (
    f_left_top @ np.ones(grid_left_top.Nphi)
) * grid_left_top.dphi  # This is now a vector, only depends on x and y
rho_right_bottom = (
    f_right_bottom @ np.ones(grid_right_bottom.Nphi)
) * grid_right_bottom.dphi  # This is now a vector, only depends on x and y
rho_right_top = (
    f_right_top @ np.ones(grid_right_top.Nphi)
) * grid_right_top.dphi  # This is now a vector, only depends on x and y

rho_matrix_left_bottom = rho_left_bottom.reshape(
    (grid_left_bottom.Nx, grid_left_bottom.Ny), order="F"
)
rho_matrix_left_top = rho_left_top.reshape(
    (grid_left_top.Nx, grid_left_top.Ny), order="F"
)
rho_matrix_right_bottom = rho_right_bottom.reshape(
    (grid_right_bottom.Nx, grid_right_bottom.Ny), order="F"
)
rho_matrix_right_top = rho_right_top.reshape(
    (grid_right_top.Nx, grid_right_top.Ny), order="F"
)

rho_matrix_left = np.concatenate((rho_matrix_left_bottom, rho_matrix_left_top), axis=1)
rho_matrix_right = np.concatenate(
    (rho_matrix_right_bottom, rho_matrix_right_top), axis=1
)

rho_matrix = np.concatenate((rho_matrix_left, rho_matrix_right), axis=0)

extent = [grid.X[0], grid.X[-1], grid.Y[0], grid.Y[-1]]

fig, axes = plt.subplots(1, 1, figsize=(10, 8))

im = axes.imshow(rho_matrix.T, extent=extent, origin="lower")
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel("$y$", fontsize=fs)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([0, 0.5, 1])
axes.tick_params(axis="both", labelsize=fs, pad=10)

cbar_fixed = fig.colorbar(im, ax=axes)
cbar_fixed.set_ticks([np.min(rho_matrix), np.max(rho_matrix)])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "dd_splitting_2x1d_rho_final.pdf")


# ### Slice plot

# rho_vector_x = rho_matrix[:,int(grid_left.Ny/2)]

# fig, axes = plt.subplots(1, 1, figsize=(10, 8))
# plt.plot(grid.X, rho_vector_x)
# plt.title("$y$ fixed")
# axes.set_xlabel("$x$", fontsize=fs)
# axes.set_ylabel("$f(x)$", fontsize=fs)
# plt.savefig(savepath + "dd_splitting_2x1d_rho_x_final.pdf")


# ### Rank plots

# fig, axes = plt.subplots(1, 1, figsize=(10, 8))
# plt.plot(time,  rank_left_bottom_adapted)
# plt.title("rank left bottom adapted")
# axes.set_xlabel("$t$", fontsize=fs)
# axes.set_ylabel("$r(t)$", fontsize=fs)
# plt.savefig(savepath + "dd_splitting_2x1d_rank_left_bottom_adapted.pdf")

# fig, axes = plt.subplots(1, 1, figsize=(10, 8))
# plt.plot(time,  rank_left_bottom_dropped)
# plt.title("rank left bottom dropped")
# axes.set_xlabel("$t$", fontsize=fs)
# axes.set_ylabel("$r(t)$", fontsize=fs)
# plt.savefig(savepath + "dd_splitting_2x1d_rank_left_bottom_dropped.pdf")

# fig, axes = plt.subplots(1, 1, figsize=(10, 8))
# plt.plot(time,  rank_left_top_adapted)
# plt.title("rank left top adapted")
# axes.set_xlabel("$t$", fontsize=fs)
# axes.set_ylabel("$r(t)$", fontsize=fs)
# plt.savefig(savepath + "dd_splitting_2x1d_rank_left_top_adapted.pdf")

# fig, axes = plt.subplots(1, 1, figsize=(10, 8))
# plt.plot(time,  rank_left_top_dropped)
# plt.title("rank left top dropped")
# axes.set_xlabel("$t$", fontsize=fs)
# axes.set_ylabel("$r(t)$", fontsize=fs)
# plt.savefig(savepath + "dd_splitting_2x1d_rank_left_top_dropped.pdf")

# fig, axes = plt.subplots(1, 1, figsize=(10, 8))
# plt.plot(time,  rank_right_bottom_adapted)
# plt.title("rank right bottom adapted")
# axes.set_xlabel("$t$", fontsize=fs)
# axes.set_ylabel("$r(t)$", fontsize=fs)
# plt.savefig(savepath + "dd_splitting_2x1d_rank_right_bottom_adapted.pdf")

# fig, axes = plt.subplots(1, 1, figsize=(10, 8))
# plt.plot(time,  rank_right_bottom_dropped)
# plt.title("rank right bottom dropped")
# axes.set_xlabel("$t$", fontsize=fs)
# axes.set_ylabel("$r(t)$", fontsize=fs)
# plt.savefig(savepath + "dd_splitting_2x1d_rank_right_bottom_dropped.pdf")

# fig, axes = plt.subplots(1, 1, figsize=(10, 8))
# plt.plot(time,  rank_right_top_adapted)
# plt.title("rank right top adapted")
# axes.set_xlabel("$t$", fontsize=fs)
# axes.set_ylabel("$r(t)$", fontsize=fs)
# plt.savefig(savepath + "dd_splitting_2x1d_rank_right_top_adapted.pdf")

# fig, axes = plt.subplots(1, 1, figsize=(10, 8))
# plt.plot(time,  rank_right_top_dropped)
# plt.title("rank right top dropped")
# axes.set_xlabel("$t$", fontsize=fs)
# axes.set_ylabel("$r(t)$", fontsize=fs)
# plt.savefig(savepath + "dd_splitting_2x1d_rank_right_top_dropped.pdf")


# ### Print frobenius error
# print(np.linalg.norm(rho0_matrix-rho_matrix, 'fro'))
