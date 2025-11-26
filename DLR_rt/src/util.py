"""
Contains helper functions.
"""

import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from scipy import sparse
from scipy.sparse import diags

from DLR_rt.src.grid import Grid_2x1d


def compute_mass(lr, grid):
    """
    Compute mass from low rank representation.

    Parameters
    ----------
    lr
        Low rank representation.
    grid
        Grid class.
    """
    K = lr.U @ lr.S
    rho = K @ np.trapezoid(lr.V, dx=grid.dmu, axis=0).T
    M = np.trapezoid(rho, dx=grid.dx, axis=0)
    return M


def computeD_cendiff_2x1d(grid: Grid_2x1d, option_dd: str = "no_dd"):
    """
    Compute centered difference matrices.

    Compute centered difference matrices for 2x1d with or without domain decmoposition.
    For option_dd = "no_dd", periodic bc are chosen. Output is DX and DY.

    Parameters
    ----------
    grid
        Grid class of subdomain
    option_dd : str
        Can be chosen either "dd", "no_dd" or "outflow".
    """
    ### Compute DX
    # Step 1: Set up cen difference matrix
    Dx = sparse.lil_matrix((grid.Nx, grid.Nx), dtype=float)
    Dx.setdiag(-1, k=-1)
    Dx.setdiag(1, k=1)

    if option_dd == "no_dd":
        Dx[0, grid.Nx - 1] = -1
        Dx[grid.Nx - 1, 0] = 1

    if option_dd == "outflow":
        Dx[0, 0] = -2
        Dx[0, 1] = 2
        Dx[grid.Nx - 1, grid.Nx - 1] = 2
        Dx[grid.Nx - 1, grid.Nx - 2] = -2

    # If option = "dd", we need to add information afterwards with inflow/outflow 
    # and cannot just impose periodic b.c.

    Ix = sparse.identity(grid.Ny, format="csr", dtype=float)

    # Step 2: Use np.kron
    DX = sparse.kron(Ix, Dx, format="csr")  # Ny × Ny blocks, each block is Dx

    ### Compute DY
    # Step 1: Set up cen difference matrix
    Dy = sparse.lil_matrix((grid.Ny, grid.Ny), dtype=float)
    Dy.setdiag(-1, k=-1)
    Dy.setdiag(1, k=1)

    if option_dd == "no_dd":
        Dy[0, grid.Ny - 1] = -1
        Dy[grid.Ny - 1, 0] = 1

    if option_dd == "outflow":
        Dy[0, 0] = -2
        Dy[0, 1] = 2
        Dy[grid.Ny - 1, grid.Ny - 1] = 2
        Dy[grid.Ny - 1, grid.Ny - 2] = -2

    # If option = "dd", we need to add information afterwards with inflow/outflow 
    # and cannot just impose periodic b.c.

    Iy = sparse.identity(grid.Nx, format="csr", dtype=float)

    # Step 2: Use np.kron
    DY = sparse.kron(Dy, Iy, format="csr")  # Nx × Nx blocks, each block is Dy

    ### Scale matrices
    DX *= 0.5 / grid.dx
    DY *= 0.5 / grid.dy

    return DX.tocsr(), DY.tocsr()

def computeD_upwind_2x1d(grid: Grid_2x1d, option_dd: str = "no_dd"):
    """
    Compute upwind difference matrices.

    Compute upwind difference matrices for 2x1d with or without domain decomposition.
    For option_dd = "no_dd", periodic bc are chosen.
    Output is DX_0 (DX-),  DX_1 (DX+), DY_0 (DY-) and DY_1 (DY+).

    Parameters
    ----------
    grid
        Grid class of subdomain
    option_dd : str
        Can be chosen either "dd", "no_dd" or "outflow".
    """
    ### Compute DX_0
    # Step 1: Set up upwind matrix
    Dx_0 = sparse.lil_matrix((grid.Nx, grid.Nx), dtype=float)
    Dx_0.setdiag(-1, k=-1)
    Dx_0.setdiag(1, k=0)

    if option_dd == "no_dd":
        Dx_0[0, grid.Nx - 1] = -1

    if option_dd == "outflow":
        Dx_0[0, 0] = -1
        Dx_0[0, 1] = 1

    # If option = "dd", we need to add information afterwards with inflow/outflow 
    # and cannot just impose periodic b.c.

    Ix = sparse.identity(grid.Ny, format="csr", dtype=float)

    # Step 2: Use np.kron
    DX_0 = sparse.kron(Ix, Dx_0, format="csr")  # Ny × Ny blocks, each block is Dx_0

    ### Compute DX_1
    # Step 1: Set up upwind matrix
    Dx_1 = sparse.lil_matrix((grid.Nx, grid.Nx), dtype=float)
    Dx_1.setdiag(-1, k=0)
    Dx_1.setdiag(1, k=1)

    if option_dd == "no_dd":
        Dx_1[grid.Nx - 1, 0] = 1

    if option_dd == "outflow":
        Dx_1[grid.Nx - 1, grid.Nx - 1] = 1
        Dx_1[grid.Nx - 1, grid.Nx - 2] = -1

    # If option = "dd", we need to add information afterwards with inflow/outflow 
    # and cannot just impose periodic b.c.

    Ix = sparse.identity(grid.Ny, format="csr", dtype=float)

    # Step 2: Use np.kron
    DX_1 = sparse.kron(Ix, Dx_1, format="csr")  # Ny × Ny blocks, each block is Dx_1

    ### Compute DY_0
    # Step 1: Set up upwind matrix
    Dy_0 = sparse.lil_matrix((grid.Ny, grid.Ny), dtype=float)
    Dy_0.setdiag(-1, k=-1)
    Dy_0.setdiag(1, k=0)

    if option_dd == "no_dd":
        Dy_0[0, grid.Ny - 1] = -1

    if option_dd == "outflow":
        Dy_0[0, 0] = -1
        Dy_0[0, 1] = 1

    # If option = "dd", we need to add information afterwards with inflow/outflow 
    # and cannot just impose periodic b.c.

    Iy = sparse.identity(grid.Nx, format="csr", dtype=float)

    # Step 2: Use np.kron
    DY_0 = sparse.kron(Dy_0, Iy, format="csr")  # Nx × Nx blocks, each block is Dy_0

    ### Compute DY_1
    # Step 1: Set up upwind matrix
    Dy_1 = sparse.lil_matrix((grid.Ny, grid.Ny), dtype=float)
    Dy_1.setdiag(-1, k=0)
    Dy_1.setdiag(1, k=1)

    if option_dd == "no_dd":
        Dy_1[grid.Ny - 1, 0] = 1

    if option_dd == "outflow":
        Dy_1[grid.Ny - 1, grid.Ny - 1] = 1
        Dy_1[grid.Ny - 1, grid.Ny - 2] = -1

    # If option = "dd", we need to add information afterwards with inflow/outflow 
    # and cannot just impose periodic b.c.

    Iy = sparse.identity(grid.Nx, format="csr", dtype=float)

    # Step 2: Use np.kron
    DY_1 = sparse.kron(Dy_1, Iy, format="csr")  # Nx × Nx blocks, each block is Dy_1

    ### Scale matrices (different scaling then cendiff)
    DX_0 *= 1 / grid.dx
    DX_1 *= 1 / grid.dx
    DY_0 *= 1 / grid.dy
    DY_1 *= 1 / grid.dy

    return DX_0.tocsr(), DX_1.tocsr(), DY_0.tocsr(), DY_1.tocsr()

def plot_rho_subgrids(subgrids, lr_on_subgrids, fs = 26, savepath = "plots/", t = 0.0, 
                      plot_option = "normal", plot_name_add = ""):
    """
    Plot rho over x and y.

    Generate a colorplot of rho for simulation done on subgrids.

    Parameters
    ----------
    subgrids
        Grid classes on subdomains.
    lr_on_subgrids
        Low rank representation on subgrids.
    fs
        Font size for plot.
    savepath
        Path to save the plot.
    t
        Time for which plot is generated.
    plot_option
        Can be "normal" or "log", decides on type of color scale.
    plot_name_add
        Additional string to add to plot names.
    """

    ### Build rho matrix
    n_split_x = subgrids[0][0].n_split_x
    n_split_y = subgrids[0][0].n_split_y

    rho_matrix_on_subgrids = []

    for j in range(n_split_y):
        row = []
        for i in range(n_split_x):

            f = (lr_on_subgrids[j][i].U @ lr_on_subgrids[j][i].S @ 
                 lr_on_subgrids[j][i].V.T)
            
            rho = (f @ np.ones(subgrids[j][i].Nphi)) * subgrids[j][i].dphi

            rho_matrix = rho.reshape((subgrids[j][i].Nx, subgrids[j][i].Ny),
                                      order="F")
            row.append(rho_matrix)
        rho_matrix_on_subgrids.append(row)
    
    rho_matrix_concatenate_y_list = []
    for i in range(n_split_x):
        rho_matrix_concatenate_y = rho_matrix_on_subgrids[0][i]
        for j in range(1,n_split_y):
            rho_matrix_concatenate_y = np.concatenate((rho_matrix_concatenate_y, 
                                                    rho_matrix_on_subgrids[j][i]), 
                                                    axis=1)
        rho_matrix_concatenate_y_list.append(rho_matrix_concatenate_y)

    rho_matrix_full = rho_matrix_concatenate_y_list[0]
    for i in range(1,n_split_x):
        rho_matrix_full = np.concatenate((rho_matrix_full, 
                                          rho_matrix_concatenate_y_list[i]), axis=0)

    ### Do the plotting
    extent = [subgrids[0][0].X[0], subgrids[0][n_split_x-1].X[-1], 
              subgrids[0][0].Y[0], subgrids[n_split_y-1][0].Y[-1]]

    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    if plot_option == "normal":
        im = axes.imshow(rho_matrix_full.T, extent=extent, origin="lower", cmap="jet")
    elif plot_option == "log":
        rho_matrix_full = np.clip(rho_matrix_full, 1e-3, None)
        im = axes.imshow(rho_matrix_full.T, extent=extent, origin="lower", 
                         norm=LogNorm(vmin=1e-3, vmax=np.max(rho_matrix_full)), 
                         cmap="jet")
    axes.set_xlabel("$x$", fontsize=fs)
    axes.set_ylabel("$y$", fontsize=fs)
    axes.set_xticks([0, 0.5, 1])
    axes.set_yticks([0, 0.5, 1])
    axes.tick_params(axis="both", labelsize=fs, pad=10)

    # --- Custom formatter for n.m × 10^{x} ---
    def sci_notation_formatter(value, pos):
        if value == 0:
            return "0"
        exponent = int(np.floor(np.log10(value)))
        coeff = value / 10**exponent
        if coeff == 1.0:
            return rf"$10^{{{exponent}}}$"
        else:
            return rf"${coeff:.2f} \cdot 10^{{{exponent}}}$"

    formatter = FuncFormatter(sci_notation_formatter)

    exponent = int(np.floor(np.log10(np.max(rho_matrix_full))))
    factor = 10 ** (exponent - 3 + 1)
    cbar_max = np.floor(np.max(rho_matrix_full) / factor) * factor

    if plot_option == "normal":
        cbar_fixed = fig.colorbar(im, ax=axes)
        cbar_fixed.set_ticks([np.min(rho_matrix_full), np.max(rho_matrix_full)])
    elif plot_option == "log":
        cbar_fixed = fig.colorbar(im, ax=axes, format=formatter)
        ticks = [1e-3, cbar_max]
        cbar_fixed.set_ticks(ticks)
        cbar_fixed.ax.minorticks_off()
    cbar_fixed.ax.tick_params(labelsize=fs)

    plt.tight_layout()
    if plot_option == "normal":
        plt.savefig(savepath + plot_name_add + "dd_splitting_2x1d_subgrids_rho_t" 
                    + str(t) + ".pdf")
    elif plot_option == "log":
        plt.savefig(savepath + plot_name_add + "dd_splitting_2x1d_subgrids_rho_t" 
                    + str(t) + "_log.pdf")
    
    plt.close()

    return

def plot_ranks_subgrids(subgrids, time, 
                        rank_on_subgrids_adapted, rank_on_subgrids_dropped, 
                        fs = 26, savepath = "plots/", option = "lattice",
                        plot_name_add = ""):
    """
    Plot ranks on subgrids.

    Generate plots for rank evolution on subgrids as well as final ranks.

    Parameters
    ----------
    subgrids
        Grid classes on subdomains.
    time
        Time vector.
    rank_on_subgrids_adapted
        Adapted ranks on subgrids (vector for each subgrid).
    rank_on_subgrids_dropped
        Dropped ranks on subgrids (vector for each subgrid).
    fs
        Font size for plot.
    savepath
        Path to save the plot.
    option
        Can be "lattice" or "hohlraum", decides on type of final rank plot.
    plot_name_add
        Additional string to add to plot names.
    """
    
    ### Plot for rank over time

    n_split_x = subgrids[0][0].n_split_x
    n_split_y = subgrids[0][0].n_split_y

    all_values = []

    for j in range(n_split_y):
        for i in range(n_split_x):
            all_values.append(rank_on_subgrids_adapted[j][i])
            all_values.append(rank_on_subgrids_dropped[j][i])

    y_min = min(np.min(v) for v in all_values) - 1
    y_max = max(np.max(v) for v in all_values) + 1
    
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    for j in range(n_split_y):
        for i in range(n_split_x):

            plt.plot(time,  rank_on_subgrids_adapted[j][i])

    axes.set_xlabel("$t$", fontsize=fs)
    axes.set_ylabel(r"$r_{\text{int}}(t)$", fontsize=fs)
    axes.set_xlim(time[0], time[-1]) # Remove extra padding: set x-limits to data range
    axes.set_ylim(y_min, y_max) 
    axes.tick_params(axis='both', which='major', labelsize=fs)
    plt.savefig(savepath + plot_name_add 
                + "dd_splitting_2x1d_subgrids_rank_adapted.pdf")

    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    for j in range(n_split_y):
        for i in range(n_split_x):

            plt.plot(time,  rank_on_subgrids_dropped[j][i])

    axes.set_xlabel("$t$", fontsize=fs)
    axes.set_ylabel("$r(t)$", fontsize=fs)
    axes.set_xlim(time[0], time[-1]) # Remove extra padding: set x-limits to data range
    axes.set_ylim(y_min, y_max) 
    axes.tick_params(axis='both', which='major', labelsize=fs)
    plt.savefig(savepath + plot_name_add 
                + "dd_splitting_2x1d_subgrids_rank_dropped.pdf")

    ### Plot for final rank

    # Example data for each cell (ny rows, nx columns)
    data = np.zeros((n_split_x, n_split_y))
    for j in range(n_split_y):
        for i in range(n_split_x):
            data[i,n_split_y-j-1] = rank_on_subgrids_adapted[j][i][-1]

    if option == "lattice":
        # Create the plot
        fig, ax = plt.subplots()
        ax.imshow(data.T, cmap='viridis')  # You can change colormap

        # Show numbers in each cell
        for i in range(n_split_x):
            for j in range(n_split_y):
                ax.text(i, j, str(int(data[i, j])), ha='center', va='center', color='w')

        # Optional: Add grid lines
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_xticks(np.arange(-0.5, n_split_x, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_split_y, 1), minor=True)
        ax.grid(which='minor', color='black', linewidth=1)
        ax.tick_params(which='minor', length=0)

    else:

        norm = colors.Normalize(vmin=np.min(data), vmax=np.max(data))
        cmap = plt.get_cmap('viridis')

        fig, ax = plt.subplots()

        for j in range(n_split_y):
            for i in range(n_split_x):
                x, y = subgrids[j][i].X[0], subgrids[j][i].Y[0]
                width = subgrids[j][i].X[-1] - x + subgrids[j][i].dx
                height = subgrids[j][i].Y[-1] - y + subgrids[j][i].dy
                rank = rank_on_subgrids_adapted[j][i][-1]

                # Get color from colormap
                color = cmap(norm(rank))

                # Draw rectangle with color
                rect = patches.Rectangle((x, y), width, height, linewidth=1, 
                                         edgecolor='black', facecolor=color)
                ax.add_patch(rect)

                # Add rank label
                ax.text(x + width/2, y + height/2, str(rank), 
                        ha='center', va='center', color='white')
        
        # Clean up axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    plt.savefig(savepath + plot_name_add 
                + "dd_splitting_2x1d_subgrids_rank_adapted_final.pdf")

    # Example data for each cell (ny rows, nx columns)
    data = np.zeros((n_split_x, n_split_y))
    for j in range(n_split_y):
        for i in range(n_split_x):
            data[i,n_split_y-j-1] = rank_on_subgrids_dropped[j][i][-1]

    if option == "lattice":
        # Create the plot
        fig, ax = plt.subplots()
        ax.imshow(data.T, cmap='viridis')  # You can change colormap

        # Show numbers in each cell
        for i in range(n_split_x):
            for j in range(n_split_y):
                ax.text(i, j, str(int(data[i, j])), ha='center', va='center', color='w')

        # Optional: Add grid lines
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_xticks(np.arange(-0.5, n_split_x, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_split_y, 1), minor=True)
        ax.grid(which='minor', color='black', linewidth=1)
        ax.tick_params(which='minor', length=0)

    else:

        norm = colors.Normalize(vmin=np.min(data), vmax=np.max(data))
        cmap = plt.get_cmap('viridis')

        fig, ax = plt.subplots()

        for j in range(n_split_y):
            for i in range(n_split_x):
                x, y = subgrids[j][i].X[0], subgrids[j][i].Y[0]
                width = subgrids[j][i].X[-1] - x + subgrids[j][i].dx
                height = subgrids[j][i].Y[-1] - y + subgrids[j][i].dy
                rank = rank_on_subgrids_dropped[j][i][-1]

                # Get color from colormap
                color = cmap(norm(rank))

                # Draw rectangle with color
                rect = patches.Rectangle((x, y), width, height, linewidth=1, 
                                         edgecolor='black', facecolor=color)
                ax.add_patch(rect)

                # Add rank label
                ax.text(x + width/2, y + height/2, str(rank), 
                        ha='center', va='center', color='white')
        
        # Clean up axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    plt.savefig(savepath + plot_name_add 
                + "dd_splitting_2x1d_subgrids_rank_dropped_final.pdf")
    plt.close()

    return

def plot_rho_onedomain(grid, lr, fs = 26, savepath = "plots/", t = 0.0, 
                      plot_name_add = ""):
    """
    Plot rho over x and y.

    Generate a colorplot of rho for simulation done on one domain.

    Parameters
    ----------
    grid
        Grid class.
    lr
        Low rank representation.
    fs
        Font size for plot.
    savepath
        Path to save the plot.
    t
        Time for which plot is generated.
    plot_name_add
        Additional string to add to plot names.
    """

    f = lr.U @ lr.S @ lr.V.T

    rho = (
        f @ np.ones(grid.Nphi)
    ) * grid.dphi  # This is now a vector, only depends on x and y

    rho_matrix = rho.reshape((grid.Nx, grid.Ny), order="F")

    extent = [grid.X[0], grid.X[-1], grid.Y[0], grid.Y[-1]]
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))

    rho_matrix = np.clip(rho_matrix, 1e-3, None)
    im = axes.imshow(rho_matrix.T, extent=extent, origin="lower", 
                    norm=LogNorm(vmin=1e-3, vmax=np.max(rho_matrix)), 
                    cmap="jet")
    axes.set_xlabel("$x$", fontsize=fs)
    axes.set_ylabel("$y$", fontsize=fs)
    axes.set_xticks([0, 0.5, 1])
    axes.set_yticks([0, 0.5, 1])
    axes.tick_params(axis="both", labelsize=fs, pad=10)

    # --- Custom formatter for n.m × 10^{x} ---
    def sci_notation_formatter(value, pos):
        if value == 0:
            return "0"
        exponent = int(np.floor(np.log10(value)))
        coeff = value / 10**exponent
        if coeff == 1.0:
            return rf"$10^{{{exponent}}}$"
        else:
            return rf"${coeff:.2f} \cdot 10^{{{exponent}}}$"

    formatter = FuncFormatter(sci_notation_formatter)

    exponent = int(np.floor(np.log10(np.max(rho_matrix))))
    factor = 10 ** (exponent - 3 + 1)
    cbar_max = np.floor(np.max(rho_matrix) / factor) * factor

    cbar_fixed = fig.colorbar(im, ax=axes, format=formatter)
    ticks = [1e-3, cbar_max]
    cbar_fixed.set_ticks(ticks)
    
    cbar_fixed.ax.minorticks_off()
    cbar_fixed.ax.tick_params(labelsize=fs)

    plt.tight_layout()
    plt.savefig(savepath + plot_name_add + "2x1d_rho_t" + str(t) + "_spacedepcoeff.pdf")
    plt.close()

    return

def generate_full_f(lr_on_subgrids, subgrids, grid):
    """
    Generate full f from low rank representations on subgrids.

    Parameters
    ----------
    lr_on_subgrids
        Low rank representations on subgrids.
    subgrids
        Grid classes on subdomains.
    grid
        Grid class for full domain.
    """

    n_split_x = subgrids[0][0].n_split_x
    n_split_y = subgrids[0][0].n_split_y

    Nx = grid.Nx
    Ny = grid.Ny
    Nphi = grid.Nphi

    f_dd = np.zeros((Nx*Ny,Nphi))

    start = 0
    for j in range(n_split_y):
        f_block = np.zeros((Nx*subgrids[j][0].Ny,Nphi))
        starting_gridpoints_x = 0
        for i in range(n_split_x):

            f = (lr_on_subgrids[j][i].U @ lr_on_subgrids[j][i].S @ 
                    lr_on_subgrids[j][i].V.T)
            
            for k in range(subgrids[j][i].Ny):
                f_block[k*Nx+starting_gridpoints_x:
                        k*Nx+starting_gridpoints_x+subgrids[j][i].Nx,
                        :] = f[k*subgrids[j][i].Nx:(k+1)*subgrids[j][i].Nx,:]

            starting_gridpoints_x += subgrids[j][i].Nx

        f_dd[start:start+subgrids[j][i].Ny * Nx,:] = f_block

        start += subgrids[j][i].Ny * Nx

    return f_dd

def setup_coeff_source_1domain(Nx, Ny, option_bc):
    """
    Setup scattering, transport and advection coefficients as well as source.

    Setup according to test case.

    Parameters
    ----------
    Nx
        Number of grid points in x.
    Ny
        Number of grid points in y.
    option_bc : str
        Can be chosen either "lattice", "hohlraum" or "pointsource".
    """

    if option_bc == "lattice":
        ### Full lattice setup
        c_adv_vec = np.ones(Nx*Ny)
        c_adv = diags(c_adv_vec)

        # Parameters
        num_blocks = 7        # number of blocks in each row/col
        block_size = int(Nx/num_blocks)        # size of each block

        # Pattern of blocks
        block_pattern_s = np.array([[1,1,1,1,1,1,1],
                                    [1,0,1,0,1,0,1],
                                    [1,1,0,1,0,1,1],
                                    [1,0,1,1,1,0,1],
                                    [1,1,0,1,0,1,1],
                                    [1,0,1,1,1,0,1],
                                    [1,1,1,1,1,1,1]])
        block_pattern_t = np.array([[1,1,1,1,1,1,1],
                                    [1,10,1,10,1,10,1],
                                    [1,1,10,1,10,1,1],
                                    [1,10,1,1,1,10,1],
                                    [1,1,10,1,10,1,1],
                                    [1,10,1,1,1,10,1],
                                    [1,1,1,1,1,1,1]])

        # Expand each block into block_size x block_size
        c_s_matrix = np.kron(block_pattern_s, np.ones((block_size, block_size),
                                                       dtype=int))
        c_t_matrix = np.kron(block_pattern_t, np.ones((block_size, block_size),
                                                       dtype=int))

        # Change to vector
        c_s_vec = c_s_matrix.flatten()
        c_t_vec = c_t_matrix.flatten()

        # Change to diag matrix
        c_s = diags(c_s_vec)
        c_t = diags(c_t_vec)

    elif option_bc == "hohlraum" or option_bc == "pointsource":
        ### Full hohlraum setup
        c_adv_vec = np.ones(Nx*Ny)
        c_adv = diags(c_adv_vec)

        c_s_matrix = np.zeros((Nx,Ny))
        c_t_matrix = np.zeros((Nx,Ny))

        # Set c_t for absorbing parts
        for i in range(Nx):
            for j in range(Ny):

                if j <= 0.05*Ny or j >= 0.95*Ny:    # upper and lower blocks
                    c_t_matrix[j,i] = 100

                else:
                    if (i >= 0.95*Nx or i<=0.05*Nx and (0.25*Ny <= j <= 0.75*Ny)
                        or (0.25*Nx <= i <= 0.75*Nx) and (0.25*Ny <= j <= 0.75*Ny)):
                        c_t_matrix[j,i] = 100

        # Change to vector
        c_s_vec = c_s_matrix.flatten()
        c_t_vec = c_t_matrix.flatten()

        # Change to diag matrix
        c_s = diags(c_s_vec)
        c_t = diags(c_t_vec)

    if option_bc == "lattice":
        ### Do normal 1 source
        # Start with all zeros
        block_matrix = np.zeros((num_blocks, num_blocks))

        # Set block (4,4) to 1
        block_row = 3
        block_col = 3
        block_matrix[block_row, block_col] = 1

        # Expand to full matrix
        source = np.kron(block_matrix, np.ones((block_size, block_size)))

    elif option_bc == "hohlraum" or option_bc == "pointsource":
        ### Hohlraum and pointsource already have inflow source
        source = np.zeros((Nx,Ny))

    return c_adv, c_s, c_t, source, c_s_matrix, c_t_matrix


def save_data_to_file(savepath, filename, lr, time, rank_int, rank):
    """
    Save data to file.

    Parameters
    ----------
    savepath
        Path to save the data.
    filename
        Name of the file.
    lr
        Low rank object to be saved.
    time
        Time vector.
    rank_int
        Intermediate rank vector.
    rank
        Rank vector.
    """

    np.savez(f"{savepath}{filename}_t{time[-1]:.4f}.npz", 
             U=lr.U, S=lr.S, V=lr.V, time=time, rank_int=rank_int, rank=rank)

    return