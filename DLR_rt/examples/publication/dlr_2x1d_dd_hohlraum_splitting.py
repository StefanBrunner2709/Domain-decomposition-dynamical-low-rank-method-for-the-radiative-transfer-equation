import matplotlib.pyplot as plt
import numpy as np

from DLR_rt.src.grid import Grid_2x1d
from DLR_rt.src.initial_condition import (
    setInitialCondition_2x1d_lr_subgrids,
)
from DLR_rt.src.lr import LR
from DLR_rt.src.run_functions import integrate_dd_hohlraum
from DLR_rt.src.util import (
    generate_full_f,
    plot_ranks_subgrids,
)


def run_dd_hohlraum(option_problem = "hohlraum", option_error_estimate = False, 
                    option_dof_plot = False):
    """
    Run the domain decomposition dynamical low rank algorithm for the 
    hohlraum/pointsource problem.

    Can be used to reproduce the results in the publication.

    Parameters
    ----------
    option_problem : str
        Problem to solve, either "hohlraum" or "pointsource" or "pointsource_2".
    option_error_estimate : bool
        Whether to compute and plot the error with respect to a higher rank
        reference solution on one domain.
    option_dof_plot : bool
        Whether to plot the degrees of freedom over time.
    """
    
    ### Plotting
    if option_problem == "hohlraum":
        Nx = 200
        Ny = 200
        t_f = 1.2
        snapshots = 7
        option_error_list = 121 if option_error_estimate else 0
        drop_tol = 1e-3
        option_problem_int = "hohlraum"  # For running code
    elif option_problem == "pointsource" or option_problem == "pointsource_2":
        Nx = 600
        Ny = 600
        t_f = 1.0
        snapshots = 11
        option_error_list = 101 if option_error_estimate else 0
        option_problem_int = "pointsource"  # For running code

    if option_problem == "pointsource":
        drop_tol = 1e-2
    elif option_problem == "pointsource_2":
        drop_tol = 1e-5

    Nphi = 200
    dt = 0.5 / Nx
    r = 5
    fs = 26
    savepath = "plots/"
    option_scheme = "upwind"
    option_rank_adaptivity = "v2"

    ### Initial configuration
    grid = Grid_2x1d(Nx, Ny, Nphi, r, _option_dd="dd")
    subgrids = grid.split_grid_into_subgrids(option_split="hohlraum")


    lr0_on_subgrids = setInitialCondition_2x1d_lr_subgrids(subgrids, 
                                                           option_cond="lattice")

    ### Final configuration
    (lr_on_subgrids, time, 
    rank_on_subgrids_adapted, rank_on_subgrids_dropped, 
    Frob_list) = integrate_dd_hohlraum(
        lr0_on_subgrids, subgrids, t_f, dt, option_scheme=option_scheme, 
        drop_tol=drop_tol, 
        option_problem=option_problem_int, snapshots=snapshots,
        plot_name_add=option_problem,
        option_rank_adaptivity=option_rank_adaptivity, 
        grid = grid, option_error_list = option_error_list
        )

    plot_ranks_subgrids(subgrids, time, rank_on_subgrids_adapted, 
                        rank_on_subgrids_dropped, option="hohlraum", 
                        plot_name_add=option_problem)

    ### Compare to higher rank solution on 1 domain
    if option_error_estimate:
        
        ### Copy data from already existing file
        data = np.load(f"data/reference_sol_{option_problem}_t{time[-1]:.4f}.npz")
        lr_2 = LR(data["U"], data["S"], data["V"])
        rank_adapted_2 = data["rank_int"]
        rank_dropped_2 = data["rank"]

        f_2 = lr_2.U @ lr_2.S @ lr_2.V.T

        f = generate_full_f(lr_on_subgrids, subgrids, grid)

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
        plt.savefig(savepath + "dd_" + option_problem + "_frobenius_error.pdf") 
        plt.close() 

    if option_dof_plot:

        ### Compute DoF for dd
        dof_int_dd_list = []
        for k in range(len(time)):
            dof_int = 0
            for j in range(5):
                for i in range(5):
                    dof_int += (rank_on_subgrids_adapted[j][i][k] * 
                            (subgrids[j][i].Nx * subgrids[j][i].Ny + 
                            subgrids[j][i].Nphi + rank_on_subgrids_adapted[j][i][k]))
            dof_int_dd_list.append(dof_int)

        dof_dd_list = []
        for k in range(len(time)):
            dof = 0
            for j in range(5):
                for i in range(5):
                    dof += (rank_on_subgrids_dropped[j][i][k] * 
                            (subgrids[j][i].Nx * subgrids[j][i].Ny + 
                            subgrids[j][i].Nphi + rank_on_subgrids_dropped[j][i][k]))
            dof_dd_list.append(dof)

        ### Compute DoF for 1 domain
        data = np.load(f"data/final_sol_{option_problem}_t{time[-1]:.4f}.npz")
        rank_adapted_2 = data["rank_int"]
        rank_dropped_2 = data["rank"]

        dof_int_1d_list = []
        for i in range(len(time)):
            dof_int = (rank_adapted_2[i] * (Nx * Ny + Nphi + rank_adapted_2[i]))
            dof_int_1d_list.append(dof_int)

        dof_1d_list = []
        for i in range(len(time)):
            dof = (rank_dropped_2[i] * (Nx * Ny + Nphi + rank_dropped_2[i]))
            dof_1d_list.append(dof)

        ### Plot DoF
        fig, axes = plt.subplots(1, 1, figsize=(10, 8))

        plt.plot(time, dof_int_dd_list, linestyle='--', color='red')
        plt.plot(time, dof_dd_list, linestyle='-',  color='red')
        plt.plot(time, dof_int_1d_list, linestyle='--', color='blue')
        plt.plot(time, dof_1d_list, linestyle='-', color='blue')

        axes.set_xlabel("$t$", fontsize=fs)
        axes.set_ylabel("DoF", fontsize=fs)
        axes.set_xlim(time[0], time[-1]) # Remove padding: set x-limits to data range
        axes.set_ylim(bottom=0)
        axes.tick_params(axis='both', which='major', labelsize=fs)

        xticks = axes.xaxis.get_major_ticks() 
        xticks[0].label1.set_visible(False)

        axes.yaxis.get_offset_text().set_fontsize(fs)

        plt.savefig(savepath + "DoF_" + option_problem + ".pdf")
        plt.close()

    return
