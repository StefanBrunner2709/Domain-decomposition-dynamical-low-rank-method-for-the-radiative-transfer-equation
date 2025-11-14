import matplotlib.pyplot as plt
import numpy as np

from DLR_rt.src.grid import Grid_2x1d
from DLR_rt.src.initial_condition import (
    setInitialCondition_2x1d_lr_subgrids,
)
from DLR_rt.src.lr import LR
from DLR_rt.src.run_functions import integrate_dd_lattice
from DLR_rt.src.util import (
    generate_full_f,
    plot_ranks_subgrids,
)

### Plotting

Nx = 252
Ny = 252
Nphi = 252
dt = 0.5 / Nx
r = 5
t_f = 0.7
snapshots = 8
fs = 26
savepath = "plots/"
method = "lie"
option_scheme = "upwind"
option_timescheme = "RK4"
option_rank_adaptivity = "v2"

option_error_estimate = True
option_error_list = 71

tol_sing_val = 2e-6
drop_tol = 2e-6
tol_lattice = 2e-6


### Initial configuration
grid = Grid_2x1d(Nx, Ny, Nphi, r, _option_dd="dd", _coeff=[1.0, 1.0, 1.0])
subgrids = grid.split_grid_into_subgrids(option_coeff="lattice", 
                                         n_split_y=7, n_split_x=7)


lr0_on_subgrids = setInitialCondition_2x1d_lr_subgrids(subgrids, option_cond="lattice")

### Final configuration
(lr_on_subgrids, time, 
 rank_on_subgrids_adapted, rank_on_subgrids_dropped, Frob_list) = integrate_dd_lattice(
    lr0_on_subgrids, subgrids, t_f, dt, 
    option_scheme=option_scheme, option_timescheme=option_timescheme,
    tol_sing_val=tol_sing_val, drop_tol=drop_tol, snapshots=snapshots, 
    option_rank_adaptivity=option_rank_adaptivity,
    grid = grid, option_error_list = option_error_list
    )

plot_ranks_subgrids(subgrids, time, rank_on_subgrids_adapted, rank_on_subgrids_dropped)


### Compare to higher rank solution on 1 domain
if option_error_estimate:
    
    ### Copy data from already existing file
    data = np.load(f"data/reference_sol_lattice_t{time[-1]:.4f}.npz")
    lr_2 = LR(data["U"], data["S"], data["V"])
    time_2 = data["time"]
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
    plt.savefig(savepath + "dd_lattice_frobenius_error.pdf")  
