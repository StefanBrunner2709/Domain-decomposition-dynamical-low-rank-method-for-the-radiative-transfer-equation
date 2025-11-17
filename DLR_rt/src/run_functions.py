"""
Contains integration functions for different example scripts
"""

import numpy as np
from tqdm import tqdm

from DLR_rt.src.grid import Grid_2x1d
from DLR_rt.src.integrators import PSI_lie, PSI_splitting_lie
from DLR_rt.src.lr import LR, computeF_b_2x1d_X, computeF_b_2x1d_Y
from DLR_rt.src.util import (
    computeD_cendiff_2x1d,
    computeD_upwind_2x1d,
    generate_full_f,
    plot_rho_onedomain,
    plot_rho_subgrids,
    save_data_to_file,
)


def integrate_dd_hohlraum(lr0_on_subgrids: LR, subgrids: Grid_2x1d, 
                          t_f: float, dt: float,
                          tol_sing_val: float = 1e-3, drop_tol: float = 1e-7, 
                          option_scheme: str = "cendiff", 
                          option_problem : str = "hohlraum", 
                          snapshots: int = 2, plot_name_add = "",
                          option_rank_adaptivity: str = "v1",
                          grid: Grid_2x1d = None, option_error_list: int = 0):
    """
    Integrate low rank structure for hohlraum setup with domain decomposition.

    Can be used for either hohlraum or pointsource simulations. Integrations is done 
    according to evolution equations obtained after splitting the radiative transfer 
    equation.

    Parameters
    ----------
    lr0_on_subgrids
        Low rank structures on subdomains at initial time.
    subgrids
        Grid classes on subdomains.
    t_f
        Final time.
    dt
        Time step size.
    tol_sing_val
        Tolerance for add basis function step.
    drop_tol
        Tolerance for drop basis function step.
    option_scheme
        Can be chosen "cendiff" or "upwind".
    option_problem
        Can be chosen "hohlraum" or "pointsource".
    snapshots
        Number of snapshots to be taken (including initial and final time).
    plot_name_add
        Additional string to add to plot names.
    option_rank_adaptivity
        Possible options are "v1" or "v2".
    grid
        Full grid class for error computation.
    option_error_list
        If > 0, compute error to reference solution.
    """
    
    lr_on_subgrids = lr0_on_subgrids
    t = 0
    time = []
    time.append(t)

    n_split_x = subgrids[0][0].n_split_x
    n_split_y = subgrids[0][0].n_split_y

    D_on_subgrids = []

    for j in range(n_split_y):
        row = []
        for i in range(n_split_x):

            DX, DY = computeD_cendiff_2x1d(subgrids[j][i], "dd")
            # all grids have same size, thus enough to compute once

            if option_scheme == "upwind":
                DX_0, DX_1, DY_0, DY_1 = computeD_upwind_2x1d(subgrids[j][i], "dd")
            else:
                DX_0 = None
                DX_1 = None
                DY_0 = None
                DY_1 = None

            D = [DX, DY, DX_0, DX_1, DY_0, DY_1]

            row.append(D)
        D_on_subgrids.append(row)

    rank_on_subgrids_adapted = []
    rank_on_subgrids_dropped = []

    for j in range(n_split_y):
        row_adapted = []
        row_dropped = []
        for i in range(n_split_x):

            rank_adapted = [subgrids[j][i].r]
            rank_dropped = [subgrids[j][i].r]

            row_adapted.append(rank_adapted)
            row_dropped.append(rank_dropped)
        rank_on_subgrids_adapted.append(row_adapted)
        rank_on_subgrids_dropped.append(row_dropped)

    # --- SNAPSHOT setup ---
    if snapshots < 2:
        snapshots = 2  # At least initial and final
    snapshot_times = [i * t_f / (snapshots - 1) for i in range(snapshots)]
    next_snapshot_idx = 1  # first snapshot after t=0

    # --- Initial snapshot ---
    print(f"📸 Snapshot 1/{snapshots} at t = {t:.4f}")
    plot_rho_subgrids(subgrids, lr_on_subgrids, t=t, plot_option="log", 
                      plot_name_add=plot_name_add)
    
    # --- Compare to reference solution ---
    Frob_list = []

    if option_error_list > 0:
        error_list_times = [i * t_f /
                             (option_error_list - 1) for i in range(option_error_list)]
        next_error_list_idx = 1

        # Copy data from already existing file
        data = np.load(f"data/reference_sol_{option_problem}_t{t:.4f}.npz")
        lr_2 = LR(data["U"], data["S"], data["V"])

        f_2 = lr_2.U @ lr_2.S @ lr_2.V.T

        f = generate_full_f(lr_on_subgrids, subgrids, grid)

        Frob = np.linalg.norm(f - f_2, ord='fro')
        Frob /= np.sqrt(grid.Nx * grid.Ny * grid.Nphi)

        Frob_list.append(Frob)

    with tqdm(total=t_f / dt, desc="Running Simulation") as pbar:
        while t < t_f:
            pbar.update(1)

            if t + dt > t_f:
                dt = t_f - t

            ### Calculate f
            f_on_subgrids = []

            for j in range(n_split_y):
                row = []
                for i in range(n_split_x):
                    
                    f = (lr_on_subgrids[j][i].U @ lr_on_subgrids[j][i].S @ 
                        lr_on_subgrids[j][i].V.T)
                    
                    row.append(f)
                f_on_subgrids.append(row)

            ### Calculate F_b
            F_b_X_on_subgrids = []
            F_b_Y_on_subgrids = []
            
            for j in range(n_split_y):
                row_F_b_X = []
                row_F_b_Y = []
                for i in range(n_split_x):

                    if i==0:
                        F_b_X = computeF_b_2x1d_X(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_left=f_on_subgrids[j][n_split_x-1],
                                                  f_right=f_on_subgrids[j][i+1],
                                                  grid_left=subgrids[j][n_split_x-1],
                                                  grid_right=subgrids[j][i+1])     
                        if option_problem == "hohlraum":
                            for k in range(len(subgrids[j][i].PHI)):  # inflow left is 1
                                if (subgrids[j][i].PHI[k] < np.pi / 2 
                                    or subgrids[j][i].PHI[k] > 3 / 2 * np.pi):
                                    F_b_X[: len(subgrids[j][i].Y), k] = 1
                        elif option_problem == "pointsource":
                            for k in range(len(subgrids[j][i].PHI)):
                                if (subgrids[j][i].PHI[k] < np.pi / 2 
                                    or subgrids[j][i].PHI[k] > 3 / 2 * np.pi):
                                    F_b_X[: len(subgrids[j][i].Y), k] = 0
                                    if j == n_split_y-2:
                                        F_b_X[: len(subgrids[j][i].Y), k] = (
                                            1
                                            / (np.sqrt(2 * np.pi)*1e-2)
                                            * np.exp(-((subgrids[j][i].Y - 0.85
                                                        -subgrids[j][i].dy/2) ** 2) 
                                                     / (2*(1e-2)**2))
                                        )
                    elif i==n_split_x-1:
                        F_b_X = computeF_b_2x1d_X(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_left=f_on_subgrids[j][i-1],
                                                  f_right=f_on_subgrids[j][0],
                                                  grid_left=subgrids[j][i-1],
                                                  grid_right=subgrids[j][0])
                        for k in range(len(subgrids[j][i].PHI)):  # set outflow boundary
                            if (subgrids[j][i].PHI[k] >= np.pi / 2 
                                and subgrids[j][i].PHI[k] <= 3 / 2 * np.pi):
                                F_b_X[len(subgrids[j][i].Y) :, k] = 0
                    else:
                        F_b_X = computeF_b_2x1d_X(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_left=f_on_subgrids[j][i-1],
                                                  f_right=f_on_subgrids[j][i+1],
                                                  grid_left=subgrids[j][i-1],
                                                  grid_right=subgrids[j][i+1])
                        
                    if j==0:
                        F_b_Y = computeF_b_2x1d_Y(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_bottom=f_on_subgrids[n_split_y-1][i],
                                                  f_top=f_on_subgrids[j+1][i],
                                                  grid_bottom=subgrids[n_split_y-1][i],
                                                  grid_top=subgrids[j+1][i])
                        for k in range(len(subgrids[j][i].PHI)):  # set outflow boundary
                            if subgrids[j][i].PHI[k] < np.pi:
                                F_b_Y[: len(subgrids[j][i].X), k] = 0
                    elif j==n_split_y-1:
                        F_b_Y = computeF_b_2x1d_Y(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_bottom=f_on_subgrids[j-1][i],
                                                  f_top=f_on_subgrids[0][i],
                                                  grid_bottom=subgrids[j-1][i],
                                                  grid_top=subgrids[0][i])
                        for k in range(len(subgrids[j][i].PHI)):  # set outflow boundary
                            if subgrids[j][i].PHI[k] >= np.pi:
                                F_b_Y[len(subgrids[j][i].X) :, k] = 0
                    else:
                        F_b_Y = computeF_b_2x1d_Y(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_bottom=f_on_subgrids[j-1][i],
                                                  f_top=f_on_subgrids[j+1][i],
                                                  grid_bottom=subgrids[j-1][i],
                                                  grid_top=subgrids[j+1][i])
                        
                    row_F_b_X.append(F_b_X)
                    row_F_b_Y.append(F_b_Y)
                F_b_X_on_subgrids.append(row_F_b_X)
                F_b_Y_on_subgrids.append(row_F_b_Y)

            ### Update lr by PSI with adaptive rank strategy
            ### Run PSI with adaptive rank strategy
            for j in range(n_split_y):
                for i in range(n_split_x):

                    # if (j==1 or j==3) and i==0:
                    #     source= np.ones((subgrids[j][i].Nx, subgrids[j][i].Ny))
                    #     source = source.flatten()[:, None]
                    # else:
                    source = None

                    (lr_on_subgrids[j][i], 
                     subgrids[j][i], 
                     rank_on_subgrids_adapted[j][i], 
                     rank_on_subgrids_dropped[j][i]) = PSI_splitting_lie(
                        lr_on_subgrids[j][i],
                        subgrids[j][i],
                        dt,
                        F_b_X_on_subgrids[j][i],
                        F_b_Y_on_subgrids[j][i],
                        DX=D_on_subgrids[j][i][0],
                        DY=D_on_subgrids[j][i][1],
                        tol_sing_val=tol_sing_val,
                        drop_tol=drop_tol,
                        rank_adapted=rank_on_subgrids_adapted[j][i],
                        rank_dropped=rank_on_subgrids_dropped[j][i],
                        source=source,
                        option_scheme=option_scheme, 
                        DX_0=D_on_subgrids[j][i][2], 
                        DX_1=D_on_subgrids[j][i][3], 
                        DY_0=D_on_subgrids[j][i][4], 
                        DY_1=D_on_subgrids[j][i][5],
                        option_rank_adaptivity=option_rank_adaptivity
                    )

            ### Update time
            t += dt
            time.append(t)

            # --- Check for snapshot condition ---
            if next_snapshot_idx < snapshots and t >= snapshot_times[next_snapshot_idx]:
                print(f"📸 Snapshot {next_snapshot_idx+1}/{snapshots} at t = {t:.4f}")
                plot_rho_subgrids(subgrids, lr_on_subgrids, t=t, plot_option="log", 
                                  plot_name_add=plot_name_add)
                next_snapshot_idx += 1

            # --- Check for error computation condition ---
            if (option_error_list > 0 and next_error_list_idx < option_error_list 
                and t >= error_list_times[next_error_list_idx]):
                
                # Copy data from already existing file
                data = np.load(f"data/reference_sol_{option_problem}_t{t:.4f}.npz")
                lr_2 = LR(data["U"], data["S"], data["V"])

                f_2 = lr_2.U @ lr_2.S @ lr_2.V.T

                f = generate_full_f(lr_on_subgrids, subgrids, grid)

                Frob = np.linalg.norm(f - f_2, ord='fro')
                Frob /= np.sqrt(grid.Nx * grid.Ny * grid.Nphi)

                Frob_list.append(Frob)

                next_error_list_idx += 1

    return (lr_on_subgrids, time, 
            rank_on_subgrids_adapted, rank_on_subgrids_dropped, Frob_list)


def integrate_dd_lattice(lr0_on_subgrids: LR, subgrids: Grid_2x1d, 
                         t_f: float, dt: float,
                         tol_sing_val: float = 1e-3, drop_tol: float = 1e-7,
                         option_scheme: str = "cendiff", 
                         option_timescheme : str = "RK4", 
                         snapshots: int = 2, plot_name_add = "",
                         option_rank_adaptivity: str = "v1",
                         grid: Grid_2x1d = None, option_error_list: int = 0):
    """
    Integrate low rank structure for lattice setup with domain decomposition.

    Can be used for lattice simulations. Integrations is done according to evolution 
    equations obtained after splitting the radiative transfer equation.

    Parameters
    ----------
    lr0_on_subgrids
        Low rank structures on subdomains at initial time.
    subgrids
        Grid classes on subdomains.
    t_f
        Final time.
    dt
        Time step size.
    tol_sing_val
        Tolerance for add basis function step.
    drop_tol
        Tolerance for drop basis function step.
    option_scheme
        Can be chosen "cendiff" or "upwind".
    option_timescheme
        Can be chosen "RK4", "impl_Euler" or "impl_Euler_gmres".
    snapshots
        Number of snapshots to be taken (including initial and final time).
    plot_name_add
        Additional string to add to plot names.
    option_rank_adaptivity
        Possible options are "v1" or "v2".
    grid
        Full grid class for error computation.
    option_error_list
        If > 0, compute error to reference solution.
    """
    
    lr_on_subgrids = lr0_on_subgrids
    t = 0
    time = []
    time.append(t)

    DX, DY = computeD_cendiff_2x1d(subgrids[0][0], "dd")
    # all grids have same size, thus enough to compute once

    if option_scheme == "upwind":
        DX_0, DX_1, DY_0, DY_1 = computeD_upwind_2x1d(subgrids[0][0], "dd")
    else:
        DX_0 = None
        DX_1 = None
        DY_0 = None
        DY_1 = None

    n_split_x = subgrids[0][0].n_split_x
    n_split_y = subgrids[0][0].n_split_y

    rank_on_subgrids_adapted = []
    rank_on_subgrids_dropped = []

    for j in range(n_split_y):
        row_adapted = []
        row_dropped = []
        for i in range(n_split_x):

            rank_adapted = [subgrids[j][i].r]
            rank_dropped = [subgrids[j][i].r]

            row_adapted.append(rank_adapted)
            row_dropped.append(rank_dropped)
        rank_on_subgrids_adapted.append(row_adapted)
        rank_on_subgrids_dropped.append(row_dropped)

    # --- SNAPSHOT setup ---
    if snapshots < 2:
        snapshots = 2  # At least initial and final
    snapshot_times = [i * t_f / (snapshots - 1) for i in range(snapshots)]
    next_snapshot_idx = 1  # first snapshot after t=0

    # --- Initial snapshot ---
    print(f"📸 Snapshot 1/{snapshots} at t = {t:.4f}")
    plot_rho_subgrids(subgrids, lr_on_subgrids, t=t, plot_option="log", 
                      plot_name_add=plot_name_add)
    
    # --- Compare to reference solution ---
    Frob_list = []

    if option_error_list > 0:
        error_list_times = [i * t_f /
                             (option_error_list - 1) for i in range(option_error_list)]
        next_error_list_idx = 1

        # Copy data from already existing file
        data = np.load(f"data/reference_sol_lattice_t{t:.4f}.npz")
        lr_2 = LR(data["U"], data["S"], data["V"])

        f_2 = lr_2.U @ lr_2.S @ lr_2.V.T

        f = generate_full_f(lr_on_subgrids, subgrids, grid)

        Frob = np.linalg.norm(f - f_2, ord='fro')
        Frob /= np.sqrt(grid.Nx * grid.Ny * grid.Nphi)

        Frob_list.append(Frob)
        
    with tqdm(total=t_f / dt, desc="Running Simulation") as pbar:
        while t < t_f:
            pbar.update(1)

            if t + dt > t_f:
                dt = t_f - t

            ### Calculate f
            f_on_subgrids = []

            for j in range(n_split_y):
                row = []
                for i in range(n_split_x):
                    
                    f = (lr_on_subgrids[j][i].U @ lr_on_subgrids[j][i].S @ 
                        lr_on_subgrids[j][i].V.T)
                    
                    row.append(f)
                f_on_subgrids.append(row)

            ### Calculate F_b
            F_b_X_on_subgrids = []
            F_b_Y_on_subgrids = []
            
            for j in range(n_split_y):
                row_F_b_X = []
                row_F_b_Y = []
                for i in range(n_split_x):

                    if i==0:
                        F_b_X = computeF_b_2x1d_X(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_left=f_on_subgrids[j][n_split_x-1],
                                                  f_right=f_on_subgrids[j][i+1])     
                        for k in range(len(subgrids[j][i].PHI)):  # set outflow boundary
                            if (subgrids[j][i].PHI[k] < np.pi / 2 
                                or subgrids[j][i].PHI[k] > 3 / 2 * np.pi):
                                F_b_X[: len(subgrids[j][i].Y), k] = 0
                    elif i==n_split_x-1:
                        F_b_X = computeF_b_2x1d_X(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_left=f_on_subgrids[j][i-1],
                                                  f_right=f_on_subgrids[j][0])
                        for k in range(len(subgrids[j][i].PHI)):  # set outflow boundary
                            if (subgrids[j][i].PHI[k] >= np.pi / 2 
                                and subgrids[j][i].PHI[k] <= 3 / 2 * np.pi):
                                F_b_X[len(subgrids[j][i].Y) :, k] = 0
                    else:
                        F_b_X = computeF_b_2x1d_X(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_left=f_on_subgrids[j][i-1],
                                                  f_right=f_on_subgrids[j][i+1])
                        
                    if j==0:
                        F_b_Y = computeF_b_2x1d_Y(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_bottom=f_on_subgrids[n_split_y-1][i],
                                                  f_top=f_on_subgrids[j+1][i])
                        for k in range(len(subgrids[j][i].PHI)):  # set outflow boundary
                            if subgrids[j][i].PHI[k] < np.pi:
                                F_b_Y[: len(subgrids[j][i].X), k] = 0
                    elif j==n_split_y-1:
                        F_b_Y = computeF_b_2x1d_Y(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_bottom=f_on_subgrids[j-1][i],
                                                  f_top=f_on_subgrids[0][i])
                        for k in range(len(subgrids[j][i].PHI)):  # set outflow boundary
                            if subgrids[j][i].PHI[k] >= np.pi:
                                F_b_Y[len(subgrids[j][i].X) :, k] = 0
                    else:
                        F_b_Y = computeF_b_2x1d_Y(f_on_subgrids[j][i],subgrids[j][i],
                                                  f_bottom=f_on_subgrids[j-1][i],
                                                  f_top=f_on_subgrids[j+1][i])
                        
                    row_F_b_X.append(F_b_X)
                    row_F_b_Y.append(F_b_Y)
                F_b_X_on_subgrids.append(row_F_b_X)
                F_b_Y_on_subgrids.append(row_F_b_Y)

            ### Update lr by PSI with adaptive rank strategy
            ### Run PSI with adaptive rank strategy
            for j in range(n_split_y):
                for i in range(n_split_x):

                    if i==3 and j==3:
                        source= np.ones((subgrids[j][i].Nx, subgrids[j][i].Ny))
                        source = source.flatten()[:, None]
                    else:
                        source = None

                    (lr_on_subgrids[j][i], 
                     subgrids[j][i], 
                     rank_on_subgrids_adapted[j][i], 
                     rank_on_subgrids_dropped[j][i]) = PSI_splitting_lie(
                        lr_on_subgrids[j][i],
                        subgrids[j][i],
                        dt,
                        F_b_X_on_subgrids[j][i],
                        F_b_Y_on_subgrids[j][i],
                        DX=DX,
                        DY=DY,
                        tol_sing_val=tol_sing_val,
                        drop_tol=drop_tol,
                        rank_adapted=rank_on_subgrids_adapted[j][i],
                        rank_dropped=rank_on_subgrids_dropped[j][i],
                        source=source,
                        option_scheme=option_scheme, 
                        DX_0=DX_0, DX_1=DX_1, DY_0=DY_0, DY_1=DY_1,
                        option_timescheme=option_timescheme,
                        option_rank_adaptivity=option_rank_adaptivity
                    )

            ### Update time
            t += dt
            time.append(t)

            # --- Check for snapshot condition ---
            if next_snapshot_idx < snapshots and t >= snapshot_times[next_snapshot_idx]:
                print(f"📸 Snapshot {next_snapshot_idx+1}/{snapshots} at t = {t:.4f}")
                plot_rho_subgrids(subgrids, lr_on_subgrids, t=t, plot_option="log", 
                                  plot_name_add=plot_name_add)
                next_snapshot_idx += 1

            # --- Check for error computation condition ---
            if (option_error_list > 0 and next_error_list_idx < option_error_list 
                and t >= error_list_times[next_error_list_idx]):
                
                # Copy data from already existing file
                data = np.load(f"data/reference_sol_lattice_t{t:.4f}.npz")
                lr_2 = LR(data["U"], data["S"], data["V"])

                f_2 = lr_2.U @ lr_2.S @ lr_2.V.T

                f = generate_full_f(lr_on_subgrids, subgrids, grid)

                Frob = np.linalg.norm(f - f_2, ord='fro')
                Frob /= np.sqrt(grid.Nx * grid.Ny * grid.Nphi)

                Frob_list.append(Frob)

                next_error_list_idx += 1

    return (lr_on_subgrids, time, 
            rank_on_subgrids_adapted, rank_on_subgrids_dropped, Frob_list)

def integrate_1domain(lr0: LR, grid: Grid_2x1d, t_f: float, dt: float, 
              option: str = "lie", source = None, 
              option_scheme : str = "cendiff", option_timescheme : str = "RK4",
              option_bc : str = "standard", tol_sing_val = 1e-2, drop_tol = 1e-3, 
              tol_lattice = 1e-5, snapshots: int = 2, plot_name_add = "",
              option_rank_adaptivity: str = "v1",
              option_data_saves: int = 0,
              option_error_list: int = 0):
    """
    Integrate low rank structure on 1 domain.

    Can be used for lattice, hohlraum and pointsource simulations. Integrations is done 
    according to evolution equations for the radiative transfer equation.

    Parameters
    ----------
    lr0
        Low rank structure at initial time.
    grid
        Grid class.
    t_f
        Final time.
    dt
        Time step size.
    option
        Can be chosen "lie".
    source
        Source term in rt equation, if given.
    option_scheme
        Can be chosen "cendiff" or "upwind".
    option_timescheme
        Can be chosen "RK4", "impl_Euler" or "impl_Euler_gmres".
    option_bc
        Can be chosen "standard", "lattice", "hohlraum" or "pointsource".
    tol_sing_val
        Tolerance for add basis function step.
    drop_tol
        Tolerance for drop basis function step.
    tol_lattice
        Tolerance for rank adaptation without inflow conditions in lattice setup.
    snapshots
        Number of snapshots to be taken (including initial and final time).
    plot_name_add
        Additional string to add to plot names.
    option_rank_adaptivity
        Possible options are "v1" or "v2".
    option_data_saves
        If > 0, number of data saves during simulation.
    option_error_list
        If > 0, compute error to reference solution.
    """
    min_rank = grid.r

    if option_bc == "lattice" or option_bc == "hohlraum" or option_bc == "pointsource":
        rank_adapted = [grid.r]
        rank_dropped = [grid.r]
    else:
        rank_adapted = None
        rank_dropped = None

    lr = lr0
    t = 0
    time = []
    time.append(t)

    DX, DY = computeD_cendiff_2x1d(grid, "dd")

    if option_scheme == "upwind":
        DX_0, DX_1, DY_0, DY_1 = computeD_upwind_2x1d(grid, "dd")
    else:
        DX_0 = None
        DX_1 = None
        DY_0 = None
        DY_1 = None

    # --- SNAPSHOT setup ---
    if snapshots >= 2:
        snapshot_times = [i * t_f / (snapshots - 1) for i in range(snapshots)]
        next_snapshot_idx = 1  # first snapshot after t=0

        # --- Initial snapshot ---
        print(f"📸 Snapshot 1/{snapshots} at t = {t:.4f}")
        plot_rho_onedomain(grid, lr, t=t, plot_name_add=plot_name_add)
        
    # --- Sava data ---
    if option_data_saves > 0:
        data_saves_times = [i * t_f /
                             (option_data_saves - 1) for i in range(option_data_saves)]
        next_data_saves_idx = 1

        save_data_to_file(savepath = "data/", 
                            filename = "reference_sol_" + option_bc, 
                            lr=lr, time=time,
                            rank_int = rank_adapted, rank = rank_dropped)
        
    # --- Compare to reference solution ---
    Frob_list = []

    if option_error_list > 0:
        error_list_times = [i * t_f /
                             (option_error_list - 1) for i in range(option_error_list)]
        next_error_list_idx = 1

        # Copy data from already existing file
        data = np.load(f"data/reference_sol_{option_bc}_t{t:.4f}.npz")
        lr_2 = LR(data["U"], data["S"], data["V"])

        f_2 = lr_2.U @ lr_2.S @ lr_2.V.T

        f = lr.U @ lr.S @ lr.V.T

        Frob = np.linalg.norm(f - f_2, ord='fro')
        Frob /= np.sqrt(grid.Nx * grid.Ny * grid.Nphi)

        Frob_list.append(Frob)

    with tqdm(total=t_f / dt, desc="Running Simulation") as pbar:
        while t < t_f:
            pbar.update(1)

            if t + dt > t_f:
                dt = t_f - t

            if (option_bc == "lattice" or option_bc == "hohlraum" 
                or option_bc == "pointsource"):

                f = lr.U @ lr.S @ lr.V.T

                F_b_X = computeF_b_2x1d_X(f, grid, option_bc = option_bc)
                F_b_Y = computeF_b_2x1d_Y(f, grid, option_bc = option_bc)

            else:
                F_b_X = None
                F_b_Y = None

            if option == "lie":
                (lr, grid, 
                 rank_adapted, rank_dropped) = PSI_lie(lr, grid, dt, DX=DX, DY=DY, 
                                   dimensions="2x1d", option_coeff="space_dep", 
                                   source=source, option_scheme=option_scheme,
                                   DX_0=DX_0, DX_1=DX_1, DY_0=DY_0, DY_1=DY_1,
                                   option_timescheme=option_timescheme,
                                   option_bc = option_bc, F_b_X = F_b_X, F_b_Y = F_b_Y,
                                   tol_sing_val=tol_sing_val, drop_tol=drop_tol, 
                                   min_rank=min_rank, 
                                   rank_adapted=rank_adapted, rank_dropped=rank_dropped,
                                   tol_lattice=tol_lattice, 
                                   option_rank_adaptivity=option_rank_adaptivity)

            t += dt
            time.append(t)

            # --- Check for snapshot condition ---
            if (snapshots >= 2 and next_snapshot_idx < snapshots and 
                t >= snapshot_times[next_snapshot_idx]):
                print(f"📸 Snapshot {next_snapshot_idx+1}/{snapshots} at t = {t:.4f}")
                plot_rho_onedomain(grid, lr, t=t, plot_name_add=plot_name_add)
                next_snapshot_idx += 1
                
            # --- Save data ---
            if (option_data_saves > 0 and next_data_saves_idx < option_data_saves 
                and t >= data_saves_times[next_data_saves_idx]):
                
                save_data_to_file(savepath = "data/", 
                                  filename = "reference_sol_" + option_bc, 
                                  lr=lr, time=time,
                                  rank_int = rank_adapted, rank = rank_dropped)
                
                next_data_saves_idx += 1

            # --- Check for error computation condition ---
            if (option_error_list > 0 and next_error_list_idx < option_error_list 
                and t >= error_list_times[next_error_list_idx]):
                
                # Copy data from already existing file
                data = np.load(f"data/reference_sol_{option_bc}_t{t:.4f}.npz")
                lr_2 = LR(data["U"], data["S"], data["V"])

                f_2 = lr_2.U @ lr_2.S @ lr_2.V.T

                f = lr.U @ lr.S @ lr.V.T

                Frob = np.linalg.norm(f - f_2, ord='fro')
                Frob /= np.sqrt(grid.Nx * grid.Ny * grid.Nphi)

                Frob_list.append(Frob)

                next_error_list_idx += 1

    return lr, time, rank_adapted, rank_dropped, Frob_list
