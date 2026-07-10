import matplotlib.pyplot as plt
import numpy as np

from DLR_rt.src.grid import Grid_2x1d
from DLR_rt.src.initial_condition import (
    setInitialCondition_2x1d_lr,
    setInitialCondition_2x1d_lr_subgrids,
)
from DLR_rt.src.run_functions import integrate_dd_linesource
from DLR_rt.src.util import (
    plot_ranks_subgrids,
)


def run_dd_linesource(option_dof_plot = False):
    """
    Run the domain decomposition dynamical low rank algorithm for the 
    line source problem.

    Can be used to reproduce the results in the publication.
    """

    ### Plotting

    Nx = 400
    Ny = 400
    Nphi = 200
    dt = 0.5 / Nx
    r = 5
    t_f = 0.5
    snapshots = 6
    fs = 26
    savepath = "plots/"
    option_scheme = "upwind"
    option_timescheme = "RK4"
    option_rank_adaptivity = "v2"

    drop_tol = 1e-6


    ### Initial configuration
    grid = Grid_2x1d(Nx, Ny, Nphi, r, _option_dd="dd", _coeff=[1.0, 1.0, 1.0])
    subgrids = grid.split_grid_into_subgrids(n_split_y=8, n_split_x=8)


    lr0_on_subgrids = setInitialCondition_2x1d_lr_subgrids(subgrids, 
                                                           option_cond="linesource")
    
    ### Setup boundary neighbouring domains (all 0 because of linesource simulation)
    n_split_y = 8
    n_split_x = 8
    grid_boundary_neigboring = Grid_2x1d(int(Nx/n_split_x), int(Ny/n_split_y), Nphi, 
                            1, _X = subgrids[0][0].X, _Y = subgrids[0][0].Y, 
                            _coeff=[1.0, 1.0, 1.0], _option_dd="dd")
    lr_boundary = setInitialCondition_2x1d_lr(grid_boundary_neigboring, 
                                              option_cond="zero")
    
    ### Final configuration
    (lr_on_subgrids, time, 
    rank_on_subgrids_adapted, rank_on_subgrids_dropped, 
    Frob_list) = integrate_dd_linesource(
        lr0_on_subgrids, subgrids, t_f, dt, 
        option_scheme=option_scheme, option_timescheme=option_timescheme,
        drop_tol=drop_tol, snapshots=snapshots,
        plot_name_add="linesource",
        option_rank_adaptivity=option_rank_adaptivity,
        grid = grid,
        lr_boundary = lr_boundary
        )

    plot_ranks_subgrids(subgrids, time, 
                        rank_on_subgrids_adapted, rank_on_subgrids_dropped,
                        option="linesource", plot_name_add="linesource")
    
    if option_dof_plot:

        ### Compute DoF for dd
        dof_int_dd_list = []
        for k in range(len(time)):
            dof_int = 0
            for j in range(n_split_y):
                for i in range(n_split_x):
                    dof_int += (rank_on_subgrids_adapted[j][i][k] * 
                            (subgrids[j][i].Nx * subgrids[j][i].Ny + 
                            subgrids[j][i].Nphi + rank_on_subgrids_adapted[j][i][k]))
            dof_int_dd_list.append(dof_int)

        dof_dd_list = []
        for k in range(len(time)):
            dof = 0
            for j in range(n_split_y):
                for i in range(n_split_x):
                    dof += (rank_on_subgrids_dropped[j][i][k] * 
                            (subgrids[j][i].Nx * subgrids[j][i].Ny + 
                            subgrids[j][i].Nphi + rank_on_subgrids_dropped[j][i][k]))
            dof_dd_list.append(dof)

        ### Compute DoF for 1 domain
        data = np.load(f"data/final_sol_linesource_t{time[-1]:.4f}.npz")
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

        plt.savefig(savepath + "DoF_linesource.pdf")
        plt.close()

        return
