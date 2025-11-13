"""
Contains functions to set initial condition.
"""

import numpy as np

from DLR_rt.src.grid import Grid_1x1d, Grid_2x1d
from DLR_rt.src.lr import LR


def setInitialCondition_1x1d_full(grid: Grid_1x1d, sigma: float = 1.0) -> np.ndarray:
    """
    Set initial condition.

    Set initial condition for 1x1d and full grid with periodic boundary conditions.

    Parameters
    ----------
    grid
        Grid class.
    sigma
        Standard deviation of Gaussian.
    """
    f0 = np.zeros((grid.Nx, grid.Nmu))
    xx = 1 / (2 * np.pi * sigma**2) * np.exp(-((grid.X - 0.5) ** 2) / (2 * sigma**2))
    vv = np.exp(-(np.abs(grid.MU) ** 2) / (16 * sigma**2))
    f0 = np.outer(xx, vv)
    return f0


def setInitialCondition_1x1d_lr(grid: Grid_1x1d, sigma: float = 1.0):
    """
    Set initial condition.

    Set initial condition 1x1d and low rank grid with periodic or inflow boundary 
    conditions.
    Boundary conditions are determined according to the boundary conditions in grid.

    Parameters
    ----------
    grid
        Grid class.
    sigma
        Standard deviation of Gaussian.
    """
    S = np.zeros((grid.r, grid.r))

    if grid.option_bc == "inflow":
        U = np.random.rand(grid.Nx, grid.r)
        V = np.random.rand(grid.Nmu, grid.r)
    elif grid.option_bc == "periodic":
        U = np.zeros((grid.Nx, grid.r))
        V = np.zeros((grid.Nmu, grid.r))
        U[:, 0] = (
            1 / (2 * np.pi * sigma**2) * np.exp(-((grid.X - 0.5) ** 2) / (2 * sigma**2))
        )
        V[:, 0] = np.exp(-(np.abs(grid.MU) ** 2) / (16 * sigma**2))
        S[0, 0] = 1.0

    U_ortho, R_U = np.linalg.qr(U, mode="reduced")
    U_ortho /= (np.sqrt(grid.dx))
    R_U *= (np.sqrt(grid.dx))

    V_ortho, R_V = np.linalg.qr(V, mode="reduced")
    V_ortho /= np.sqrt(grid.dmu)
    R_V *= np.sqrt(grid.dmu)

    S_ortho = R_U @ S @ R_V.T

    lr = LR(U_ortho, S_ortho, V_ortho)
    return lr


def setInitialCondition_2x1d_lr(grid: Grid_2x1d, option_cond: str = "standard"):
    """
    Set initial condition.

    Set initial condition for 2x1d low rank grid without domain decomposition.
    Set option_cond = "lattice" for almost 0 initial condition.

    Parameters
    ----------
    grid
        Grid class.
    option_cond
        Can be chosen "standard", "lattice" or "f_direct".
    """
    S = np.zeros((grid.r, grid.r))
    U = np.zeros((grid.Nx * grid.Ny, grid.r))
    V = np.zeros((grid.Nphi, grid.r))

    if option_cond == "standard":

        for i in range(grid.Ny):
            U[i * grid.Nx : (i + 1) * grid.Nx, 0] = (
                1
                / (2 * np.pi)
                * np.exp(-((grid.X - 0.5) ** 2) / 0.02)
                * np.exp(-((grid.Y[i] - 0.5) ** 2) / 0.02)
            )
            # U[i*grid.Nx:(i+1)*grid.Nx, 0] = (
            #     np.sin(2*np.pi*grid.X)*np.sin(2*np.pi*grid.Y[i])
            # )
        V[:,0] = 1.0 / grid.Nphi
        S[0, 0] = 1.0
    
    elif option_cond == "lattice":
        U[:,0] = 1e-9
        # V[8, 0] = 1.0
        V[:,0] = 1.0 / grid.Nphi
        S[0,0] = 1.0

    elif option_cond == "f_direct":
        f = np.zeros((grid.Nx * grid.Ny, grid.Nphi))
        for i in range(grid.Ny):
            f[i * grid.Nx : (i + 1) * grid.Nx, 0] = (
                1
                / (2 * np.pi)
                * np.exp(-((grid.X - 0.5) ** 2) / 0.02)
                * np.exp(-((grid.Y[i] - 0.5) ** 2) / 0.02)
            )

        U, S, Vt = np.linalg.svd(f, full_matrices=False)

        U /= (np.sqrt(grid.dx) * np.sqrt(grid.dy))
        Vt /= np.sqrt(grid.dphi)
        S *= np.sqrt(grid.dx * grid.dy * grid.dphi)

        V = Vt.T

        U = U[:,:grid.r]
        V = V[:,:grid.r]
        S = np.diag(S[:grid.r])

    U_ortho, R_U = np.linalg.qr(U, mode="reduced")
    U_ortho /= (np.sqrt(grid.dx) * np.sqrt(grid.dy))
    R_U *= (np.sqrt(grid.dx) * np.sqrt(grid.dy))

    V_ortho, R_V = np.linalg.qr(V, mode="reduced")
    V_ortho /= np.sqrt(grid.dphi)
    R_V *= np.sqrt(grid.dphi)

    S_ortho = R_U @ S @ R_V.T

    lr = LR(U_ortho, S_ortho, V_ortho)
    return lr


def setInitialCondition_2x1d_lr_subgrids(subgrids, option_cond: str = "standard"):
    """
    Set initial condition.

    Set initial condition for 2x1d low rank simulation with domain decomposition 
    on subgrids.
    Set option_cond = "lattice" for almost 0 initial condition.

    Parameters
    ----------
    subgrids
        Grid classes on subdomains.
    option_cond
        Can be chosen "standard" or "lattice".
    """

    n_split_x = subgrids[0][0].n_split_x
    n_split_y = subgrids[0][0].n_split_y

    lr_on_subgrids = []

    for j in range(n_split_y):
        row = []
        for i in range(n_split_x):

            S = np.zeros((subgrids[j][i].r, subgrids[j][i].r))
            U = np.zeros((subgrids[j][i].Nx * subgrids[j][i].Ny, subgrids[j][i].r))
            V = np.zeros((subgrids[j][i].Nphi, subgrids[j][i].r))

            if option_cond == "standard":
                for k in range(subgrids[j][i].Ny):
                    U[k * subgrids[j][i].Nx : (k + 1) * subgrids[j][i].Nx, 0] = (
                        1
                        / (2 * np.pi)
                        * np.exp(-((subgrids[j][i].X - 0.5) ** 2) / 0.02)
                        * np.exp(-((subgrids[j][i].Y[k] - 0.5) ** 2) / 0.02)
                    )
                V[0, 0] = 1.0
                S[0, 0] = 1.0

            elif option_cond == "lattice":
                U[:,0] = 1e-9
                #V[0, 0] = 1.0
                V[:,0] = 1.0 / subgrids[j][i].Nphi
                S[0,0] = 1.0

            U_ortho, R_U = np.linalg.qr(U, mode="reduced")
            U_ortho /= (np.sqrt(subgrids[j][i].dx) * np.sqrt(subgrids[j][i].dy))
            R_U *= (np.sqrt(subgrids[j][i].dx) * np.sqrt(subgrids[j][i].dy))

            V_ortho, R_V = np.linalg.qr(V, mode="reduced")
            V_ortho /= np.sqrt(subgrids[j][i].dphi)
            R_V *= np.sqrt(subgrids[j][i].dphi)

            S_ortho = R_U @ S @ R_V.T

            lr = LR(U_ortho, S_ortho, V_ortho)

            row.append(lr)
        lr_on_subgrids.append(row)

    return lr_on_subgrids
