import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from DLR_rt.src.grid import Grid_1x1d
from DLR_rt.src.initial_condition import setInitialCondition_1x1d_full
from DLR_rt.src.integrators import RK4

### Implementation of solver


def integrate(
    f0: np.ndarray,
    grid: Grid_1x1d,
    t_f: float,
    dt: float,
    epsilon: float,
    option: str,
    tol: float = 1e-2,
    tol2: float = 1e-4,
):
    f = np.copy(f0)
    t = 0
    time = [0]
    rank = []
    rank2 = []
    rank.append(np.linalg.matrix_rank(f, tol))
    rank2.append(np.linalg.matrix_rank(f, tol2))

    with tqdm(total=t_f / dt, desc="Running Simulation") as pbar:
        while t < t_f:
            pbar.update(1)

            if t + dt > t_f:
                dt = t_f - t

            if option == "upwind":  # For upwind use Euler
                f = f + dt * rhs(f, grid, epsilon, option)
                t += dt

            elif option == "cen_diff":  # For cen_diff use RK4
                f += dt * RK4(f, lambda f: rhs(f, grid, epsilon, option), dt)
                t += dt

            time.append(t)
            rank.append(np.linalg.matrix_rank(f, tol))
            rank2.append(np.linalg.matrix_rank(f, tol2))

            # ### Write values to file
            # if np.round(t * 1000) % 50 == 0:
            #     time_string = str(np.round(t * 1000))
            #     path = (
            #         "C:/Users/brunn/OneDrive/Dokumente/00_Uni/"
            #         "Masterarbeit/PHD_project_master_thesis/"
            #         "Plots_250418/Plots_classical/solution_matrices/"
            #     )
            #     np.save(
            #         path
            #         + option
            #         + "_"
            #         + time_string
            #         + ".npy",
            #         f,
            #     )

    return f, time, rank, rank2


### Implementation of cen_diff and upwind


def rhs(f: np.ndarray, grid: Grid_1x1d, epsilon: float, option: str):
    # integrate over mu to get rho
    rho = np.zeros((grid.Nx, grid.Nmu))
    rho[:] = (1 / np.sqrt(2)) * np.trapezoid(f, grid.MU, axis=1)

    # do cen diff and rest
    res = np.zeros((grid.Nx, grid.Nmu))
    if option == "cen_diff":
        for k in range(0, grid.Nmu):
            for l in range(1, grid.Nx - 1):  # noqa: E741
                res[l, k] = -(1 / epsilon) * grid.MU[k] * (
                    f[l + 1, k] - f[l - 1, k]
                ) / (2 * grid.dx) + (1 / epsilon**2) * (
                    (1 / np.sqrt(2)) * rho[l, k] - f[l, k]
                )

            res[0, k] = -(1 / epsilon) * grid.MU[k] * (f[1, k] - f[grid.Nx - 1, k]) / (
                2 * grid.dx
            ) + (1 / epsilon**2) * ((1 / np.sqrt(2)) * rho[0, k] - f[0, k])
            res[grid.Nx - 1, k] = -(1 / epsilon) * grid.MU[k] * (
                f[0, k] - f[grid.Nx - 2, k]
            ) / (2 * grid.dx) + (1 / epsilon**2) * (
                (1 / np.sqrt(2)) * rho[grid.Nx - 1, k] - f[grid.Nx - 1, k]
            )
    elif option == "upwind":
        for k in range(0, grid.Nmu):
            if grid.MU[k] >= 0:
                for l in range(1, grid.Nx):  # noqa: E741
                    res[l, k] = -(1 / epsilon) * grid.MU[k] * (
                        f[l, k] - f[l - 1, k]
                    ) / (grid.dx) + (1 / epsilon**2) * (
                        (1 / np.sqrt(2)) * rho[l, k] - f[l, k]
                    )
                    res[0, k] = -(1 / epsilon) * grid.MU[k] * (
                        f[0, k] - f[grid.Nx - 1, k]
                    ) / (grid.dx) + (1 / epsilon**2) * (
                        (1 / np.sqrt(2)) * rho[0, k] - f[0, k]
                    )
            elif grid.MU[k] < 0:
                for l in range(0, grid.Nx - 1):  # noqa: E741
                    res[l, k] = -(1 / epsilon) * grid.MU[k] * (
                        f[l + 1, k] - f[l, k]
                    ) / (grid.dx) + (1 / epsilon**2) * (
                        (1 / np.sqrt(2)) * rho[l, k] - f[l, k]
                    )
                    res[grid.Nx - 1, k] = -(1 / epsilon) * grid.MU[k] * (
                        f[0, k] - f[grid.Nx - 1, k]
                    ) / (grid.dx) + (1 / epsilon**2) * (
                        (1 / np.sqrt(2)) * rho[grid.Nx - 1, k] - f[grid.Nx - 1, k]
                    )
    return res


### Plotting

fs = 16
n = 64
t_final = 0.1
t_string = "t03"
sigma = 1.0
savepath = "plots/"


### Inital condition plot
grid = Grid_1x1d(n, n, _option_bc="periodic")
extent = [grid.X[0], grid.X[-1], grid.MU[0], grid.MU[-1]]
f0 = setInitialCondition_1x1d_full(grid, sigma)
plt.imshow(f0.T, extent=extent, origin="lower", aspect=0.5, vmin=0.13, vmax=0.16)
cbar = plt.colorbar()
cbar.set_ticks([0.13, 0.145, 0.16])
cbar.ax.tick_params(labelsize=fs)
plt.xlabel("$x$", fontsize=fs)
plt.ylabel(r"$\mu$", fontsize=fs)
plt.xticks([0, 0.5, 1], fontsize=fs)
plt.yticks([-1, 0, 1], fontsize=fs)
plt.tick_params(axis="x", pad=10)  # Moves x-axis labels farther
plt.tick_params(axis="y", pad=10)  # Moves y-axis labels farther
plt.title(r"$f(t=0,x,\mu)$", fontsize=fs)
plt.tight_layout()
plt.savefig(savepath + "init_cond_sigma" + str(sigma) + ".pdf")


### Distribution function f for different times using centered differences

f1_all = integrate(f0, grid, t_final, 1e-3, 1, "cen_diff")
f1 = f1_all[0]

plt.figure()
plt.imshow(f1.T, extent=extent, origin="lower", aspect=0.5, vmin=0.13, vmax=0.16)
cbar_fixed = plt.colorbar()
cbar_fixed.set_ticks([0.13, 0.145, 0.16])
cbar_fixed.ax.tick_params(labelsize=fs)
plt.xlabel("$x$", fontsize=fs)
plt.ylabel(r"$\mu$", fontsize=fs)
plt.xticks([0, 0.5, 1], fontsize=fs)
plt.yticks([-1, 0, 1], fontsize=fs)
plt.tick_params(axis="x", pad=10)
plt.tick_params(axis="y", pad=10)
plt.title("RK4, centered differences", fontsize=fs)
plt.tight_layout()
plt.savefig(
    savepath
    + "distr_funct_"
    + t_string
    + "_cendiff_sigma"
    + str(sigma)
    + "_fixedaxis.pdf"
)

plt.figure()
plt.imshow(f1.T, extent=extent, origin="lower", aspect=0.5)
cbar_f1 = plt.colorbar()
cbar_f1.set_ticks(
    [np.ceil(np.min(f1) * 10000) / 10000, np.floor(np.max(f1) * 10000) / 10000]
)
cbar_f1.ax.tick_params(labelsize=fs)
plt.xlabel("$x$", fontsize=fs)
plt.ylabel(r"$\mu$", fontsize=fs)
plt.xticks([0, 0.5, 1], fontsize=fs)
plt.yticks([-1, 0, 1], fontsize=fs)
plt.tick_params(axis="x", pad=10)
plt.tick_params(axis="y", pad=10)
plt.title("RK4, centered differences", fontsize=fs)
plt.tight_layout()
plt.savefig(
    savepath + "distr_funct_" + t_string + "_cendiff_sigma" + str(sigma) + ".pdf"
)


# ### Distribution function f for different times using upwind

# f2_all = integrate(f0, grid, t_final, 1e-3, 1, "upwind")
# f2 = f2_all[0]

# plt.figure()
# plt.imshow(f2.T, extent=extent, origin="lower", aspect=0.5, vmin=0.13, vmax=0.16)
# cbar_fixed = plt.colorbar()
# cbar_fixed.set_ticks([0.13, 0.145, 0.16])
# cbar_fixed.ax.tick_params(labelsize=fs)
# plt.xlabel("$x$", fontsize=fs)
# plt.ylabel(r"$\mu$", fontsize=fs)
# plt.xticks([0, 0.5, 1], fontsize=fs)
# plt.yticks([-1, 0, 1], fontsize=fs)
# plt.tick_params(axis="x", pad=10)
# plt.tick_params(axis="y", pad=10)
# plt.title("Explicit Euler, upwind", fontsize=fs)
# plt.tight_layout()
# plt.savefig(
#     savepath
#     + "distr_funct_"
#     + t_string
#     + "_upwind_sigma"
#     + str(sigma)
#     + "_fixedaxis.pdf"
# )

# plt.figure()
# plt.imshow(f2.T, extent=extent, origin="lower", aspect=0.5)
# cbar_f1 = plt.colorbar()
# cbar_f1.set_ticks(
#     [np.ceil(np.min(f2) * 10000) / 10000, np.floor(np.max(f2) * 1000) / 1000]
# )
# cbar_f1.ax.tick_params(labelsize=fs)
# plt.xlabel("$x$", fontsize=fs)
# plt.ylabel(r"$\mu$", fontsize=fs)
# plt.xticks([0, 0.5, 1], fontsize=fs)
# plt.yticks([-1, 0, 1], fontsize=fs)
# plt.tick_params(axis="x", pad=10)
# plt.tick_params(axis="y", pad=10)
# plt.title("Explicit Euler, upwind", fontsize=fs)
# plt.tight_layout()
# plt.savefig(
#     savepath + "distr_funct_" + t_string + "_upwind_sigma" + str(sigma) + ".pdf"
# )


# ### Rank plot for multiple times

# fig, ax = plt.subplots()
# ax.plot(f1_all[1], f1_all[2])
# ax.set_xlabel("$t$", fontsize=fs)
# ax.set_ylabel("rank $r(t)$", fontsize=fs)
# ax.tick_params(axis="both", labelsize=fs)
# ax.set_yticks([2, 4, 6, 8, 10, 12, 14])
# ax.margins(x=0)
# plt.tight_layout()
# plt.savefig(savepath + "rank_over_time_cendiff_sigma" + str(sigma) + "_tol1e-2.pdf")

# fig, ax = plt.subplots()
# ax.plot(f1_all[1], f1_all[3])
# ax.set_xlabel("$t$", fontsize=fs)
# ax.set_ylabel("rank $r(t)$", fontsize=fs)
# ax.tick_params(axis="both", labelsize=fs)
# plt.tight_layout()
# plt.savefig(savepath + "rank_over_time_cendiff_sigma" + str(sigma) + "_tol1e-4.pdf")

# fig, ax = plt.subplots()
# ax.plot(f2_all[1], f2_all[2])
# ax.set_xlabel("$t$", fontsize=fs)
# ax.set_ylabel("rank $r(t)$", fontsize=fs)
# ax.tick_params(axis="both", labelsize=fs)
# plt.tight_layout()
# plt.savefig(savepath + "rank_over_time_upwind_sigma" + str(sigma) + "_tol1e-2.pdf")

# fig, ax = plt.subplots()
# ax.plot(f2_all[1], f2_all[3])
# ax.set_xlabel("$t$", fontsize=fs)
# ax.set_ylabel("rank $r(t)$", fontsize=fs)
# ax.tick_params(axis="both", labelsize=fs)
# plt.tight_layout()
# plt.savefig(savepath + "rank_over_time_upwind_sigma" + str(sigma) + "_tol1e-4.pdf")
