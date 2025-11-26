import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from DLR_rt.src.grid import Grid_1x1d
from DLR_rt.src.initial_condition import setInitialCondition_1x1d_lr
from DLR_rt.src.integrators import PSI_lie, PSI_strang
from DLR_rt.src.lr import LR
from DLR_rt.src.util import compute_mass


def integrate(lr0: LR, grid: Grid_1x1d, t_f: float, dt: float, option: str = "lie"):
    lr = lr0
    t = 0
    time = []
    time.append(t)
    mass_array = []  # only needed for mass plots
    mass_initial = compute_mass(lr, grid)
    mass_array.append(0)

    with tqdm(total=t_f / dt, desc="Running Simulation") as pbar:
        while t < t_f:
            pbar.update(1)

            if t + dt > t_f:
                dt = t_f - t

            if option == "lie":
                lr, grid = PSI_lie(lr, grid, dt)

            if option == "strang":
                lr, grid = PSI_strang(lr, grid, dt, t)

            t += dt
            time.append(t)

            mass_array.append(
                (compute_mass(lr, grid) - mass_initial) / mass_initial
            )  # again only for mass plots

    return lr, time, mass_array


### Plotting

Nx = 64
Nmu = 64
dt = 1e-3
r = 16
t_f = 0.5
t_string = "01"
sigma = 1
fs = 16
savepath = "plots/"
method = "strang"


### Just one plot for certain rank and certain time

fig, axes = plt.subplots(1, 1, figsize=(10, 8))

grid = Grid_1x1d(Nx, Nmu, r, _option_bc="periodic")
lr0 = setInitialCondition_1x1d_lr(grid, sigma)
f0 = lr0.U @ lr0.S @ lr0.V.T
lr, time, mass = integrate(lr0, grid, t_f, dt, method)
f = lr.U @ lr.S @ lr.V.T
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

im = axes.imshow(f.T, extent=extent, origin="lower", aspect=0.5)
axes.set_xlabel("$x$", fontsize=fs)
axes.set_ylabel(r"$\mu$", fontsize=fs, labelpad=-5)
axes.set_xticks([0, 0.5, 1])
axes.set_yticks([-1, 0, 1])
axes.tick_params(axis="both", labelsize=fs, pad=10)

cbar_fixed = fig.colorbar(im, ax=axes)
cbar_fixed.set_ticks([np.min(f), np.max(f)])
cbar_fixed.ax.tick_params(labelsize=fs)

plt.tight_layout()
plt.savefig(savepath + "distr_funct_t" + t_string + "_" + method + "_1e-3_r16.pdf")

# ### 4 plots, same time, different ranks

# r_array = [4, 8, 16, 32]

# fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# grid = Grid_1x1d(Nx, Nmu, r_array[0], _option_bc="periodic")
# lr0 = setInitialCondition_1x1d_lr(grid, sigma)
# f0 = lr0.U @ lr0.S @ lr0.V.T
# lr, time, mass = integrate(lr0, grid, t_f, dt, method)
# f = lr.U @ lr.S @ lr.V.T
# extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

# im1 = axes[0, 0].imshow(
#     f.T, extent=extent, origin="lower", aspect=0.5, vmin=0.143, vmax=0.155
# )
# axes[0, 0].set_title("$r=$" + str(r_array[0]), fontsize=fs)
# axes[0, 0].set_xlabel("$x$", fontsize=fs)
# axes[0, 0].set_ylabel(r"$\mu$", fontsize=fs, labelpad=-5)
# axes[0, 0].set_xticks([0, 0.5, 1])
# axes[0, 0].set_yticks([-1, 0, 1])
# axes[0, 0].tick_params(axis="both", labelsize=fs, pad=10)

# grid = Grid_1x1d(Nx, Nmu, r_array[1], _option_bc="periodic")
# lr0 = setInitialCondition_1x1d_lr(grid, sigma)
# f0 = lr0.U @ lr0.S @ lr0.V.T
# lr, time, mass = integrate(lr0, grid, t_f, dt, method)
# f = lr.U @ lr.S @ lr.V.T
# extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

# im2 = axes[0, 1].imshow(
#     f.T, extent=extent, origin="lower", aspect=0.5, vmin=0.143, vmax=0.155
# )
# axes[0, 1].set_title("$r=$" + str(r_array[1]), fontsize=fs)
# axes[0, 1].set_xlabel("$x$", fontsize=fs)
# axes[0, 1].set_ylabel(r"$\mu$", fontsize=fs, labelpad=-5)
# axes[0, 1].set_xticks([0, 0.5, 1])
# axes[0, 1].set_yticks([-1, 0, 1])
# axes[0, 1].tick_params(axis="both", labelsize=fs, pad=10)

# grid = Grid_1x1d(Nx, Nmu, r_array[2], _option_bc="periodic")
# lr0 = setInitialCondition_1x1d_lr(grid, sigma)
# f0 = lr0.U @ lr0.S @ lr0.V.T
# lr, time, mass = integrate(lr0, grid, t_f, dt, method)
# f = lr.U @ lr.S @ lr.V.T
# extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

# im3 = axes[1, 0].imshow(
#     f.T, extent=extent, origin="lower", aspect=0.5, vmin=0.143, vmax=0.155
# )
# axes[1, 0].set_title("$r=$" + str(r_array[2]), fontsize=fs)
# axes[1, 0].set_xlabel("$x$", fontsize=fs)
# axes[1, 0].set_ylabel(r"$\mu$", fontsize=fs, labelpad=-5)
# axes[1, 0].set_xticks([0, 0.5, 1])
# axes[1, 0].set_yticks([-1, 0, 1])
# axes[1, 0].tick_params(axis="both", labelsize=fs, pad=10)

# grid = Grid_1x1d(Nx, Nmu, r_array[3], _option_bc="periodic")
# lr0 = setInitialCondition_1x1d_lr(grid, sigma)
# f0 = lr0.U @ lr0.S @ lr0.V.T
# lr, time, mass = integrate(lr0, grid, t_f, dt, method)
# f = lr.U @ lr.S @ lr.V.T
# extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

# im4 = axes[1, 1].imshow(
#     f.T, extent=extent, origin="lower", aspect=0.5, vmin=0.143, vmax=0.155
# )
# axes[1, 1].set_title("$r=$" + str(r_array[3]), fontsize=fs)
# axes[1, 1].set_xlabel("$x$", fontsize=fs)
# axes[1, 1].set_ylabel(r"$\mu$", fontsize=fs, labelpad=-5)
# axes[1, 1].set_xticks([0, 0.5, 1])
# axes[1, 1].set_yticks([-1, 0, 1])
# axes[1, 1].tick_params(axis="both", labelsize=fs, pad=10)

# cbar_fixed1 = fig.colorbar(im1, ax=axes[0, 0])
# cbar_fixed1.set_ticks([0.144, 0.149, 0.154])
# cbar_fixed1.ax.tick_params(labelsize=fs)
# cbar_fixed2 = fig.colorbar(im2, ax=axes[0, 1])
# cbar_fixed2.set_ticks([0.144, 0.149, 0.154])
# cbar_fixed2.ax.tick_params(labelsize=fs)
# cbar_fixed3 = fig.colorbar(im3, ax=axes[1, 0])
# cbar_fixed3.set_ticks([0.144, 0.149, 0.154])
# cbar_fixed3.ax.tick_params(labelsize=fs)
# cbar_fixed4 = fig.colorbar(im4, ax=axes[1, 1])
# cbar_fixed4.set_ticks([0.144, 0.149, 0.154])
# cbar_fixed4.ax.tick_params(labelsize=fs)

# plt.tight_layout()
# plt.subplots_adjust(wspace=0.3, hspace=0.6)
# plt.savefig(
#     savepath + "distr_funct_different_ranks_t" + t_string + "_" + method + "_1e-3.pdf"
# )


# ### 4 plots, same rank, different times

# t_f_array = [0.5, 1.0, 2.0, 3.0]

# fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# grid = Grid_1x1d(Nx, Nmu, r, _option_bc="periodic")
# lr0 = setInitialCondition_1x1d_lr(grid, sigma)
# f0 = lr0.U @ lr0.S @ lr0.V.T
# lr, time, mass = integrate(lr0, grid, t_f_array[0], dt, method)
# f = lr.U @ lr.S @ lr.V.T
# extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

# im1 = axes[0, 0].imshow(
#     f.T, extent=extent, origin="lower", aspect=0.5, vmin=0.138, vmax=0.158
# )
# axes[0, 0].set_title("$t=$" + str(t_f_array[0]), fontsize=fs)
# axes[0, 0].set_xlabel("$x$", fontsize=fs)
# axes[0, 0].set_ylabel(r"$\mu$", fontsize=fs, labelpad=-5)
# axes[0, 0].set_xticks([0, 0.5, 1])
# axes[0, 0].set_yticks([-1, 0, 1])
# axes[0, 0].tick_params(axis="both", labelsize=fs, pad=10)

# grid = Grid_1x1d(Nx, Nmu, r, _option_bc="periodic")
# lr0 = setInitialCondition_1x1d_lr(grid, sigma)
# f0 = lr0.U @ lr0.S @ lr0.V.T
# lr, time, mass = integrate(lr0, grid, t_f_array[1], dt, method)
# f = lr.U @ lr.S @ lr.V.T
# extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

# im2 = axes[0, 1].imshow(
#     f.T, extent=extent, origin="lower", aspect=0.5, vmin=0.143, vmax=0.155
# )
# axes[0, 1].set_title("$t=$" + str(t_f_array[1]), fontsize=fs)
# axes[0, 1].set_xlabel("$x$", fontsize=fs)
# axes[0, 1].set_ylabel(r"$\mu$", fontsize=fs, labelpad=-5)
# axes[0, 1].set_xticks([0, 0.5, 1])
# axes[0, 1].set_yticks([-1, 0, 1])
# axes[0, 1].tick_params(axis="both", labelsize=fs, pad=10)

# grid = Grid_1x1d(Nx, Nmu, r, _option_bc="periodic")
# lr0 = setInitialCondition_1x1d_lr(grid, sigma)
# f0 = lr0.U @ lr0.S @ lr0.V.T
# lr, time, mass = integrate(lr0, grid, t_f_array[2], dt, method)
# f = lr.U @ lr.S @ lr.V.T
# extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

# im3 = axes[1, 0].imshow(
#     f.T, extent=extent, origin="lower", aspect=0.5, vmin=0.1484, vmax=0.1526
# )
# axes[1, 0].set_title("$t=$" + str(t_f_array[2]), fontsize=fs)
# axes[1, 0].set_xlabel("$x$", fontsize=fs)
# axes[1, 0].set_ylabel(r"$\mu$", fontsize=fs, labelpad=-5)
# axes[1, 0].set_xticks([0, 0.5, 1])
# axes[1, 0].set_yticks([-1, 0, 1])
# axes[1, 0].tick_params(axis="both", labelsize=fs, pad=10)

# grid = Grid_1x1d(Nx, Nmu, r, _option_bc="periodic")
# lr0 = setInitialCondition_1x1d_lr(grid, sigma)
# f0 = lr0.U @ lr0.S @ lr0.V.T
# lr, time, mass = integrate(lr0, grid, t_f_array[3], dt, method)
# f = lr.U @ lr.S @ lr.V.T
# extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

# im4 = axes[1, 1].imshow(
#     f.T, extent=extent, origin="lower", aspect=0.5, vmin=0.1505, vmax=0.1521
# )
# axes[1, 1].set_title("$t=$" + str(t_f_array[3]), fontsize=fs)
# axes[1, 1].set_xlabel("$x$", fontsize=fs)
# axes[1, 1].set_ylabel(r"$\mu$", fontsize=fs, labelpad=-5)
# axes[1, 1].set_xticks([0, 0.5, 1])
# axes[1, 1].set_yticks([-1, 0, 1])
# axes[1, 1].tick_params(axis="both", labelsize=fs, pad=10)

# cbar_fixed1 = fig.colorbar(im1, ax=axes[0, 0])
# cbar_fixed1.set_ticks([0.139, 0.148, 0.157])
# cbar_fixed1.ax.tick_params(labelsize=fs)
# cbar_fixed2 = fig.colorbar(im2, ax=axes[0, 1])
# cbar_fixed2.set_ticks([0.144, 0.149, 0.154])
# cbar_fixed2.ax.tick_params(labelsize=fs)
# cbar_fixed3 = fig.colorbar(im3, ax=axes[1, 0])
# cbar_fixed3.set_ticks([0.149, 0.152])
# cbar_fixed3.ax.tick_params(labelsize=fs)
# cbar_fixed4 = fig.colorbar(im4, ax=axes[1, 1])
# cbar_fixed4.set_ticks([0.151, 0.152])
# cbar_fixed4.ax.tick_params(labelsize=fs)

# plt.tight_layout()
# plt.subplots_adjust(wspace=0.3, hspace=0.6)
# plt.savefig(
#     savepath + "distr_funct_different_times_r" + str(r) + "_" + method + "_1e-3.pdf"
# )


# ### Plot mass over time

# grid = Grid_1x1d(Nx, Nmu, r, _option_bc="periodic")
# lr0 = setInitialCondition_1x1d_lr(grid, sigma)
# f0 = lr0.U @ lr0.S @ lr0.V.T
# lr, time, mass = integrate(lr0, grid, t_f, dt, method)
# f = lr.U @ lr.S @ lr.V.T
# extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]

# fig, ax = plt.subplots(figsize=(16, 8))
# ax.plot(time, mass)
# ax.set_xlabel("$t$", fontsize=26)
# ax.set_ylabel(r"$\frac{m(t)-m(0)}{m(0)}$", fontsize=32, labelpad=20)
# ax.tick_params(axis="both", labelsize=26)
# ax.set_yticks([0, 0.002, 0.004])
# ax.margins(x=0)
# plt.tight_layout()

# plt.savefig(
#     savepath
#     + "mass_over_time_"
#     + method
#     + "_1e-3_r"
#     + str(r)
#     + "_sigma"
#     + str(sigma)
#     + ".pdf"
# )

# ### Values for colorbar sigma 8e-2

# im1 = axes[0, 0].imshow(
#     f.T, extent=extent, origin="lower", aspect=0.5, vmin=0, vmax=17.2
# )
# im2 = axes[0, 1].imshow(
#     f.T, extent=extent, origin="lower", aspect=0.5, vmin=0, vmax=11.5
# )
# im3 = axes[1, 0].imshow(
#     f.T, extent=extent, origin="lower", aspect=0.5, vmin=0.91, vmax=5.17
# )
# im4 = axes[1, 1].imshow(
#     f.T, extent=extent, origin="lower", aspect=0.5, vmin=1.2, vmax=2.81
# )

# cbar_fixed1 = fig.colorbar(im1, ax=axes[0, 0])
# cbar_fixed1.set_ticks([0, 8, 16])
# cbar_fixed1.ax.tick_params(labelsize=fs)
# cbar_fixed2 = fig.colorbar(im2, ax=axes[0, 1])
# cbar_fixed2.set_ticks([0, 5.5, 11])
# cbar_fixed2.ax.tick_params(labelsize=fs)
# cbar_fixed3 = fig.colorbar(im3, ax=axes[1, 0])
# cbar_fixed3.set_ticks([1, 3, 5])
# cbar_fixed3.ax.tick_params(labelsize=fs)
# cbar_fixed4 = fig.colorbar(im4, ax=axes[1, 1])
# cbar_fixed4.set_ticks([1.2, 2.0, 2.8])
# cbar_fixed4.ax.tick_params(labelsize=fs)
