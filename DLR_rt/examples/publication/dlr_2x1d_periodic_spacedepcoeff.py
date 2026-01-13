import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from DLR_rt.src.grid import Grid_2x1d
from DLR_rt.src.initial_condition import setInitialCondition_2x1d_lr
from DLR_rt.src.lr import LR
from DLR_rt.src.run_functions import integrate_1domain
from DLR_rt.src.util import setup_coeff_source_1domain


def run_1d(option_problem = "hohlraum", option_calculate_ref = False, 
           option_error_estimate = False):
    """
    Run the classic dynamical low rank algorithm for the 
    lattice/hohlraum/pointsource problem.

    Can be used to reproduce the results in the publication.

    Parameters
    ----------
    option_problem : str
        Problem to solve, either "lattice" or "hohlraum" or "pointsource".
    option_calculate_ref : bool
        Whether to calculate and save a higher rank reference solution.    
    option_error_estimate : bool
        Whether to compute and plot the error with respect to a higher rank
        reference solution on one domain.
    """

    ### Plotting
    if option_problem == "lattice":
        Nx = 252
        Ny = 252
        Nphi = 252
        t_f = 0.7
        if option_calculate_ref:
            drop_tol = 1e-10
            tol_lattice = 1e-10
            snapshots = 0
        else:
            drop_tol = 3e-5
            tol_lattice = 3e-5
            snapshots = 8
        option_data_saves = 71 if option_calculate_ref else 0
        option_error_list = 71 if option_error_estimate else 0
    elif option_problem == "hohlraum":
        Nx = 200
        Ny = 200
        Nphi = 200
        t_f = 1.2
        if option_calculate_ref:
            drop_tol = 1e-10
            tol_lattice = 1e-10
            snapshots = 0
        else:
            drop_tol = 1e-4
            tol_lattice = 3e-5
            snapshots = 7
        option_data_saves = 121 if option_calculate_ref else 0
        option_error_list = 121 if option_error_estimate else 0
    elif option_problem == "pointsource":
        Nx = 600
        Ny = 600
        Nphi = 200
        t_f = 1.0
        if option_calculate_ref:
            drop_tol = 1e-10
            tol_lattice = 1e-10
            snapshots = 0
        else:
            drop_tol = 1e-5
            tol_lattice = 3e-5
            snapshots = 11
        option_data_saves = 101 if option_calculate_ref else 0
        option_error_list = 101 if option_error_estimate else 0

    option_grid = "dd"      # Just changes how gridpoints are chosen
    option_scheme = "upwind"
    option_timescheme = "RK4"
    option_rank_adaptivity = "v2"
    option_save_last = True

    r = 5
    dt = 0.5 / Nx
    fs = 26
    savepath = "plots/"


    ### Setup coefficients and source
    c_adv, c_s, c_t, source, _, c_t_matrix = setup_coeff_source_1domain(Nx, Ny, 
                                                                        option_problem)

    ### Setup grid and initial condition
    grid = Grid_2x1d(Nx, Ny, Nphi, r, _option_dd=option_grid, _coeff=[c_adv, c_s, c_t])
    lr0 = setInitialCondition_2x1d_lr(grid, option_cond="lattice")

    ### Plot lattice
    extent = [grid.X[0], grid.X[-1], grid.Y[0], grid.Y[-1]]
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))

    axes.imshow(c_t_matrix, extent=extent, origin="lower", cmap="jet", 
                    interpolation="none")
    axes.set_xlabel("$x$", fontsize=fs)
    axes.set_ylabel("$y$", fontsize=fs)
    axes.set_xticks([0, 0.5, 1])
    axes.set_yticks([0, 0.5, 1])
    axes.tick_params(axis="both", labelsize=fs, pad=10)

    cmap = matplotlib.colormaps.get_cmap('jet')

    if option_problem == "lattice":
        x_edges = np.linspace(extent[0], extent[1], 8)
        y_edges = np.linspace(extent[2], extent[3], 8)
        blue_patch = mpatches.Patch(color=cmap(0.0), label=r'$c_{\text{s}}=1$'  
                                    + '\n' +  r'$c_{\text{t}}=1$')
        red_patch = mpatches.Patch(color=cmap(1.0), label=r'$c_{\text{s}}=0$'  
                                + '\n' +  r'$c_{\text{t}}=10$')

    else:
        x_edges = [0, 0.05, 0.25, 0.75, 0.95, 1]
        y_edges = [0, 0.05, 0.25, 0.75, 0.95, 1]
        blue_patch = mpatches.Patch(color=cmap(0.0), label=r'$c_{\text{s}}=0$'  
                                    + '\n' +  r'$c_{\text{t}}=0$')
        red_patch = mpatches.Patch(color=cmap(1.0), label=r'$c_{\text{s}}=0$'  
                                + '\n' +  r'$c_{\text{t}}=100$')

    for x in x_edges:
        axes.axvline(x=x, color='white', linewidth=0.5)
    for y in y_edges:
        axes.axhline(y=y, color='white', linewidth=0.5)

    axes.set_axisbelow(False)

    axes.legend(
        handles=[blue_patch, red_patch],
        loc='upper left',           # anchor legend to the left side of the bbox
        bbox_to_anchor=(1.05, 1.0),  # (x, y) — place it just outside the axes
        fontsize=fs,
        labelspacing=1.0,
    )

    plt.tight_layout()
    plt.savefig(savepath + "setup_" + option_problem + ".pdf", bbox_inches="tight")
    plt.close()

    ### Plot source
    extent = [grid.X[0], grid.X[-1], grid.Y[0], grid.Y[-1]]
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))

    axes.imshow(source, extent=extent, origin="lower", cmap="viridis", 
                    interpolation="none")
    axes.set_xlabel("$x$", fontsize=fs)
    axes.set_ylabel("$y$", fontsize=fs)
    axes.set_xticks([0, 0.5, 1])
    axes.set_yticks([0, 0.5, 1])
    axes.tick_params(axis="both", labelsize=fs, pad=10)

    cmap = matplotlib.colormaps.get_cmap('viridis')

    if option_problem == "lattice":
        purple_patch = mpatches.Patch(color=cmap(0.0), label=r'$Q(x,y)=0$')
        yellow_patch = mpatches.Patch(color=cmap(1.0), label=r'$Q(x,y)=1$')

        axes.legend(
            handles=[purple_patch, yellow_patch],
            loc='upper left',           # anchor legend to the left side of the bbox
            bbox_to_anchor=(1.05, 1.0),  # (x, y) — place it just outside the axes
            fontsize=fs,
            labelspacing=1.0,
        )

    plt.tight_layout()
    plt.savefig(savepath + "source_" + option_problem + ".pdf", bbox_inches="tight")
    plt.close()

    # Prepare source for code
    source = source.flatten()[:, None]


    ### Run code and do the plotting
    lr, time, rank_adapted, rank_dropped, Frob_list = integrate_1domain(lr0, grid, 
                                                            t_f, dt, source=source, 
                        option_scheme=option_scheme, 
                        option_timescheme=option_timescheme,
                        option_bc=option_problem, drop_tol=drop_tol, 
                        tol_lattice=tol_lattice, snapshots=snapshots, 
                        plot_name_add=option_problem,
                        option_rank_adaptivity=option_rank_adaptivity,
                        option_data_saves=option_data_saves, 
                        option_error_list=option_error_list)


    ### Plot for rank over time
    if not option_calculate_ref:
        y_min = min(np.min(rank_adapted), np.min(rank_dropped)) - 1
        y_max = max(np.max(rank_adapted), np.max(rank_dropped)) + 1

        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        plt.plot(time, rank_adapted)
        axes.set_xlabel("$t$", fontsize=fs)
        axes.set_ylabel(r"$r_{\text{int}}(t)$", fontsize=fs)
        axes.set_xlim(time[0], time[-1]) # Remove padding: set x-limits to data range
        axes.set_ylim(y_min, y_max)
        axes.tick_params(axis='both', which='major', labelsize=fs)
        plt.savefig(savepath + option_problem + "_1domainsim_rank_adapted.pdf")
        plt.close()

        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        plt.plot(time, rank_dropped)
        axes.set_xlabel("$t$", fontsize=fs)
        axes.set_ylabel("$r(t)$", fontsize=fs)
        axes.set_xlim(time[0], time[-1]) # Remove padding: set x-limits to data range
        axes.set_ylim(y_min, y_max)
        axes.tick_params(axis='both', which='major', labelsize=fs)  
        plt.savefig(savepath + option_problem + "_1domainsim_rank_dropped.pdf")
        plt.close()


    ### Compare to higher rank solution on 1 domain
    if option_error_estimate:
        
        ### Copy data from already existing file
        data = np.load(f"data/reference_sol_{option_problem}_t{time[-1]:.4f}.npz")
        lr_2 = LR(data["U"], data["S"], data["V"])

        f = lr.U @ lr.S @ lr.V.T

        f_2 = lr_2.U @ lr_2.S @ lr_2.V.T

        Frob = np.linalg.norm(f - f_2, ord='fro')
        Frob /= np.sqrt(Nx * Ny * Nphi)

        print("Frobenius: ", Frob)

        ### Plot error over time

        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        time_list = np.linspace(0, time[-1], len(Frob_list))

        plt.plot(time_list, np.concatenate(([np.nan],Frob_list[1:])))

        plt.yscale("log")
        axes.set_xlabel("$t$", fontsize=fs)
        axes.set_ylabel(r"$\left\| f - f_{\text{ref}} \right\|$", fontsize=fs)

        axes.set_xlim(time_list[0], time_list[-1]) # Remove extra padding
        idx = np.linspace(0, len(time_list) - 1, 5).astype(int)
        axes.set_xticks(time_list[idx])
        axes.tick_params(axis='both', which='major', labelsize=fs)
        plt.tight_layout()
        plt.savefig(savepath + "1d_" + option_problem + "_frobenius_error.pdf")  
        plt.close()

    if option_save_last:
        np.savez(f"data/final_sol_{option_problem}_t{time[-1]:.4f}.npz", 
                U=lr.U, S=lr.S, V=lr.V, time=time, 
                rank_int=rank_adapted, rank=rank_dropped)

    return
