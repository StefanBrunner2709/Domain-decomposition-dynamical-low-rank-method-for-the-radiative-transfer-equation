import numpy as np

from DLR_rt.src.grid import Grid_2x1d
from DLR_rt.src.initial_condition import (
    setInitialCondition_2x1d_lr,
    setInitialCondition_2x1d_lr_subgrids,
)
from DLR_rt.src.lr import LR
from DLR_rt.src.run_functions import integrate_dd_lattice
from DLR_rt.src.util import (
    generate_full_f,
    plot_ranks_subgrids,
    setup_coeff_source_1domain,
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

tol_sing_val = 1e-8
drop_tol = 2e-12
tol_lattice = 1e-11


### Initial configuration
grid = Grid_2x1d(Nx, Ny, Nphi, r, _option_dd="dd", _coeff=[1.0, 1.0, 1.0])
subgrids = grid.split_grid_into_subgrids(option_coeff="lattice", 
                                         n_split_y=7, n_split_x=7)


lr0_on_subgrids = setInitialCondition_2x1d_lr_subgrids(subgrids, option_cond="lattice")

### Final configuration
(lr_on_subgrids, time, 
 rank_on_subgrids_adapted, rank_on_subgrids_dropped) = integrate_dd_lattice(
    lr0_on_subgrids, subgrids, t_f, dt, 
    option_scheme=option_scheme, option_timescheme=option_timescheme,
    tol_sing_val=tol_sing_val, drop_tol=drop_tol, snapshots=snapshots, 
    option_rank_adaptivity=option_rank_adaptivity
    )

plot_ranks_subgrids(subgrids, time, rank_on_subgrids_adapted, rank_on_subgrids_dropped)


### Compare to higher rank solution on 1 domain
if option_error_estimate:

    ### Setup coefficients and source
    (c_adv, c_s, c_t, source, 
     c_s_matrix, c_t_matrix) = setup_coeff_source_1domain(Nx, Ny, "lattice")
    # Prepare source for code
    source = source.flatten()[:, None]

    ### Setup grid and initial condition
    grid_2 = Grid_2x1d(Nx, Ny, Nphi, r, _option_dd="dd", 
                       _coeff=[c_adv, c_s, c_t])
    lr0_2 = setInitialCondition_2x1d_lr(grid_2, option_cond="lattice")
    f0_2 = lr0_2.U @ lr0_2.S @ lr0_2.V.T

    # ### Run code and do the plotting
    # lr_2, time_2, rank_adapted_2, rank_dropped_2 = integrate_1domain(
    #                     lr0_2, grid_2, t_f, dt, source=source, 
    #                     option_scheme=option_scheme, 
    #                     option_timescheme=option_timescheme,
    #                     option_bc="lattice", tol_sing_val=tol_sing_val*0.001, 
    #                     drop_tol=drop_tol*0.001, 
    #                     tol_lattice=tol_lattice*0.001, snapshots=snapshots,
    #                     plot_name_add = "high_rank_")
    
    ### Copy data from already existing file
    data = np.load("data/reference_sol_lattice_t" + str(time[-1]) + ".npz")
    lr_2 = LR(data["U"], data["S"], data["V"])
    time_2 = data["time"]
    rank_adapted_2 = data["rank_int"]
    rank_dropped_2 = data["rank"]

    f_2 = lr_2.U @ lr_2.S @ lr_2.V.T

    f = generate_full_f(lr_on_subgrids, subgrids, grid)

    Frob = np.linalg.norm(f - f_2, ord='fro')
    Frob /= np.sqrt(Nx * Ny * Nphi)

    print("Frobenius: ", Frob)
