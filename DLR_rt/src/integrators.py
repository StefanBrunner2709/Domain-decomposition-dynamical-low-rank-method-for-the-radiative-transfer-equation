"""
Contains different time integrators.
"""

import numpy as np
from scipy.optimize import root
from scipy.sparse.linalg import LinearOperator, gmres

from DLR_rt.src.lr import (
    Kstep,
    Kstep1,
    Kstep2,
    Kstep3,
    Lstep,
    Lstep1,
    Lstep2,
    Lstep3,
    Sstep,
    Sstep1,
    Sstep2,
    Sstep3,
    add_basis_functions,
    add_basis_functions_v2,
    computeB,
    computeC,
    computeD,
    computeE,
    computeF_b,
    computeF_b_2x1d_X,
    drop_basis_functions,
    rank_adaptivity_PSI,
)


def RK4(f, rhs, dt):
    """
    Runge Kutta 4.

    Time integration method.

    Parameters
    ----------
    f
        Matrix that needs to be integrated over time.
    rhs
        Right-hand side of function, given as lambda function.
    dt
        Time step size.
    """
    b_coeff = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])

    k_coeff0 = rhs(f)
    k_coeff1 = rhs(f + dt * 0.5 * k_coeff0)
    k_coeff2 = rhs(f + dt * 0.5 * k_coeff1)
    k_coeff3 = rhs(f + dt * k_coeff2)

    return (
        b_coeff[0] * k_coeff0
        + b_coeff[1] * k_coeff1
        + b_coeff[2] * k_coeff2
        + b_coeff[3] * k_coeff3
    )

def expl_Euler(f, rhs):
    """
    Explicit Euler.

    Time integration method.

    Parameters
    ----------
    f
        Matrix that needs to be integrated over time.
    rhs
        Right-hand side of function, given as lambda function.
    """

    return rhs(f)

def help_func_impl_Euler(x, f, rhs, dt):
    """
    Helper function for impl_Euler.
    """

    x = x.reshape(f.shape)

    res = x-f-dt*rhs(x)

    return res.ravel()

def impl_Euler(f, rhs, dt, option = "impl_Euler"):
    """
    Implicit Euler.

    Time integration method.
    Using either krylov subspace method or GMRES.

    Parameters
    ----------
    f
        Matrix that needs to be integrated over time.
    rhs
        Right-hand side of function, given as lambda function.
    dt
        Time step size.
    option
        Can be chosen "impl_Euler" or "impl_Euler_gmres".
    """
    
    if option == "impl_Euler":
        sol = root(help_func_impl_Euler, f.ravel(), args=(f, rhs, dt), method="krylov")
        x = sol.x

    elif option == "impl_Euler_gmres":
        n = f.size

        def mv(v):

            v = v.reshape(f.shape)
            res = v-dt*rhs(v)

            return res.ravel()

        A = LinearOperator((n, n), matvec=mv)

        x, exitCode = gmres(A, f.ravel())

        if exitCode != 0:
            print("GMRES did not fully converge, info =", exitCode)

    return x.reshape(f.shape)


def PSI_lie(lr, grid, dt, F_b=None, DX=None, DY=None, dimensions="1x1d", 
            option_coeff="constant", source=None, option_scheme="cendiff",
            DX_0=None, DX_1=None, DY_0=None, DY_1=None, option_timescheme="RK4",
            option_bc="standard", F_b_X=None, F_b_Y=None,
            tol_sing_val=None, drop_tol=None, min_rank=5,
            rank_adapted=None, rank_dropped=None, tol_lattice=None,
            option_rank_adaptivity="v1"):
    """
    Projector splitting integrator with lie splitting.

    In 1x1d, to run periodic simulations, leave the standard value F_b = None.
    In 1x1d, to run inflow simulations, set F_b.
    For higher dimensional simulations set i.e. dimensions = "2x1d"

    Parameters
    ----------
    lr
        Low rank class of subdomain.
    grid
        Grid class of subdomain.
    dt
        Time step size.
    F_b
        Boundary condition matrix for inflow conditions in 1x1d.
    DX
        Centered difference matrix in x.
    DY
        Centered difference matrix in y.
    dimensions
        Can be chosen "1x1d" or "2x1d".
    option_coeff
        Can be chosen "constant" or "space_dep".
    source
        Source term in rt equation, if given.
    option_scheme
        Can be chosen "cendiff" or "upwind".
    DX_0
        Upwind difference matrix in x (DX-).
    DX_1
        Upwind difference matrix in x (DX+).
    DY_0
        Upwind difference matrix in y (DY-).
    DY_1
        Upwind difference matrix in y (DY+).
    option_timescheme
        Can be chosen "RK4", "impl_Euler" or "impl_Euler_gmres".
    option_bc
        Set "lattice", "hohlraum" or "pointsource" for 1 domain simulations of examples.
    F_b_X
        Boundary condition matrix for X for 1 domain simulations.
    F_b_Y
        Boundary condition matrix for Y for 1 domain simulations.
    tol_sing_val
        Tolerance when adding basis functions.
    drop_tol
        Tolerance when removing basis functions.
    min_rank
        Minimum rank to be kept during rank adaptivity.
    rank_adapted
        Array of adapted ranks until this time.
    rank_dropped
        Array of dropped ranks until this time.
    tol_lattice
        Tolerance for rank adaptivity, when no inflow is given.
    option_rank_adaptivity
        Possible options are "v1" or "v2".
    """
    inflow = F_b is not None

    # Add basis functions
    if option_bc == "hohlraum" or option_bc == "pointsource":
        if option_rank_adaptivity == "v1":
            lr, grid = add_basis_functions(
                lr, grid, F_b_X, tol_sing_val, dimensions="2x1d"
            )
        else:
            lr, grid = add_basis_functions_v2(
                lr, grid, F_b_X, drop_tol
            )
    if option_bc == "lattice":
        lr, grid = rank_adaptivity_PSI(lr, grid, tol=tol_lattice, min_rank=min_rank)
    if (option_bc == "lattice" or option_bc == "hohlraum" 
        or option_bc == "pointsource"):
        rank_adapted.append(grid.r)

    # K step
    C1, C2 = computeC(lr, grid, dimensions=dimensions)
    K = lr.U @ lr.S
    if option_timescheme == "RK4":
        K += dt * RK4(
            K,
            lambda K: Kstep(
                K, C1, C2, grid, lr, F_b, DX=DX, DY=DY, inflow=inflow, 
                dimensions=dimensions, option_coeff=option_coeff, source=source,
                option_scheme=option_scheme, DX_0=DX_0, DX_1=DX_1, DY_0=DY_0, DY_1=DY_1,
                option_bc=option_bc, F_b_X=F_b_X, F_b_Y=F_b_Y
            ),
            dt,
        )   # we use RK4 for both cendiff and upwind
    elif option_timescheme == "impl_Euler" or option_timescheme == "impl_Euler_gmres":
        K = impl_Euler(
            K,
            lambda K: Kstep(
                K, C1, C2, grid, lr, F_b, DX=DX, DY=DY, inflow=inflow, 
                dimensions=dimensions, option_coeff=option_coeff, source=source,
                option_scheme=option_scheme, DX_0=DX_0, DX_1=DX_1, DY_0=DY_0, DY_1=DY_1,
                option_bc=option_bc, F_b_X=F_b_X, F_b_Y=F_b_Y
            ),
            dt,
            option = option_timescheme,
        )
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    if dimensions == "1x1d":
        lr.U /= np.sqrt(grid.dx)
        lr.S *= np.sqrt(grid.dx)
    elif dimensions == "2x1d":
        lr.U /= (np.sqrt(grid.dx) * np.sqrt(grid.dy))
        lr.S *= (np.sqrt(grid.dx) * np.sqrt(grid.dy))

    # S step
    if option_bc == "standard":
        D1 = computeD(lr, grid, F_b, DX=DX, DY=DY, 
                    dimensions=dimensions, option_coeff=option_coeff)
    elif (option_bc == "lattice" or option_bc == "hohlraum" 
          or option_bc == "pointsource"):
        D1 = computeD(lr, grid, F_b_X, F_b_Y, DX=DX, DY=DY, 
                    dimensions=dimensions, option_dd = "dd", option_coeff=option_coeff)
        
    if option_coeff == "constant":
        E1 = None
    elif option_coeff == "space_dep":
        E1 = computeE(lr, grid)
    if option_timescheme == "RK4":
        lr.S += dt * RK4(
            lr.S, lambda S: Sstep(S, C1, C2, D1, grid, 
                                inflow, dimensions=dimensions, 
                                option_coeff=option_coeff, E1=E1, source=source, 
                                lr=lr, option_bc=option_bc), dt
        )   # we use RK4 for both cendiff and upwind
    elif option_timescheme == "impl_Euler" or option_timescheme == "impl_Euler_gmres":
        lr.S = impl_Euler(
                lr.S,
                lambda S: Sstep(S, C1, C2, D1, grid, 
                                inflow, dimensions=dimensions, 
                                option_coeff=option_coeff, E1=E1, source=source, 
                                lr=lr, option_bc=option_bc),
                dt,
                option = option_timescheme,
        )

    # L step
    L = lr.V @ lr.S.T
    B1 = computeB(L, grid, dimensions=dimensions)
    if option_timescheme == "RK4":
        L += dt * RK4(
            L, lambda L: Lstep(L, D1, B1, grid, lr, inflow, dimensions=dimensions, 
                                option_coeff=option_coeff, E1=E1, source=source, 
                                option_bc=option_bc), dt
        )   # we use RK4 for both cendiff and upwind
    elif option_timescheme == "impl_Euler" or option_timescheme == "impl_Euler_gmres":
        L = impl_Euler(
            L,
            lambda L: Lstep(L, D1, B1, grid, lr, inflow, dimensions=dimensions, 
                                option_coeff=option_coeff, E1=E1, source=source, 
                                option_bc=option_bc),
            dt,
            option = option_timescheme,
        )
    lr.V, St = np.linalg.qr(L, mode="reduced")
    lr.S = St.T
    if dimensions == "1x1d":
        lr.V /= np.sqrt(grid.dmu)
        lr.S *= np.sqrt(grid.dmu)
    elif dimensions == "2x1d":
        lr.V /= np.sqrt(grid.dphi)
        lr.S *= np.sqrt(grid.dphi)

    # Drop basis for adaptive rank strategy:
    if option_bc == "hohlraum" or option_bc == "pointsource":
        lr, grid = drop_basis_functions(lr, grid, drop_tol, min_rank=min_rank, 
                                        dimensions="2x1d")
    if option_bc == "lattice" or option_bc == "hohlraum" or option_bc == "pointsource":
        rank_dropped.append(grid.r)

    return lr, grid, rank_adapted, rank_dropped


def PSI_strang(lr, grid, dt, t, F_b=None, DX=None, DY=None):
    """
    Old version of Projector splitting integrator with strang splitting.

    Not used anymore, only kept for further development.
    """
    inflow = F_b is not None

    # 1/2 K step
    C1, C2 = computeC(lr, grid)
    K = lr.U @ lr.S
    K += (
        0.5
        * dt
        * RK4(K, lambda K: Kstep(K, C1, C2, grid, lr, F_b, inflow=inflow), 0.5 * dt)
    )
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= np.sqrt(grid.dx)
    lr.S *= np.sqrt(grid.dx)

    # 1/2 S step
    D1 = computeD(lr, grid, F_b, DX=DX, DY=DY)
    lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep(S, C1, C2, D1, grid, inflow), 0.5 * dt)

    # L step
    L = lr.V @ lr.S.T
    B1 = computeB(L, grid)
    L += dt * RK4(L, lambda L: Lstep(L, D1, B1, grid, lr, inflow), dt)
    lr.V, St = np.linalg.qr(L, mode="reduced")
    lr.S = St.T
    lr.V /= np.sqrt(grid.dmu)
    lr.S *= np.sqrt(grid.dmu)

    if inflow:
        # Compute F_b
        F_b = computeF_b(
            t + 0.5 * dt, lr.U @ lr.S @ lr.V.T, grid
        )  # recalculate F_b at time t + 0.5 dt
        D1 = computeD(
            lr, grid, F_b, DX=DX, DY=DY
        )  # recalculate D1 because we recalculated F_b

    # 1/2 S step
    C1, C2 = computeC(
        lr, grid
    )  # need to recalculate C1 and C2 because we changed V in L step
    lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep(S, C1, C2, D1, grid, inflow), 0.5 * dt)

    # 1/2 K step
    K = lr.U @ lr.S
    K += (
        0.5
        * dt
        * RK4(K, lambda K: Kstep(K, C1, C2, grid, lr, F_b, inflow=inflow), 0.5 * dt)
    )
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= np.sqrt(grid.dx)
    lr.S *= np.sqrt(grid.dx)

    return lr, grid


def PSI_splitting_lie(
    lr,
    grid,
    dt,
    F_b,
    F_b_top_bottom,
    DX=None,
    DY=None,
    tol_sing_val=1e-6,
    drop_tol=1e-6,
    rank_adapted=None,
    rank_dropped=None,
    source=None,
    option_scheme="cendiff", 
    DX_0=None, 
    DX_1=None, 
    DY_0=None, 
    DY_1=None,
    option_timescheme="RK4",
    option_rank_adaptivity="v1",
):
    """
    Projector splitting integrator with equation splitting and lie splitting.

    For simulations in 2x1d and DD, together with a splitting approach.

    Parameters
    ----------
    lr
        LR class of subdomain.
    grid
        Grid class of subdomain.
    dt
        Time step size.
    F_b
        Boundary condition matrix for inflow conditions in x.
    F_b_top_bottom
        Boundary condition matrix for inflow conditions in y.
    DX
        Centered difference matrix in x of subdomain.
    DY
        Centered difference matrix in y of subdomain.
    tol_sing_val
        Tolerance when adding basis functions.
    drop_tol
        Tolerance when removing basis functions.
    rank_adapted
        Array of adapted ranks until this time.
    rank_dropped
        Array of dropped ranks until this time.
    source
        Source term in rt equation, if given.
    option_scheme
        Possible options are "cendiff" or "upwind".
    DX_0
        Upwind difference matrix in x (DX-) of subdomain.
    DX_1
        Upwind difference matrix in x (DX+) of subdomain.
    DY_0
        Upwind difference matrix in y (DY-) of subdomain.
    DY_1
        Upwind difference matrix in y (DY+) of subdomain.
    option_timescheme
        Possible options are "RK4", "impl_Euler" or "impl_Euler_gmres".
    option_rank_adaptivity
        Possible options are "v1" or "v2".
    """

    # Step 1: advection in x

    ### Add basis for adaptive rank strategy:
    if option_rank_adaptivity == "v1":
        lr, grid = add_basis_functions(
            lr, grid, F_b, tol_sing_val, dimensions="2x1d"
        )
    else:
        lr, grid = add_basis_functions_v2(
            lr, grid, F_b, drop_tol
        )


    if rank_adapted is not None:
        rank_adapted.append(grid.r)

    # K step
    C1, C2 = computeC(lr, grid, dimensions="2x1d")
    K = lr.U @ lr.S
    if option_timescheme == "RK4":
        K += dt * RK4(K, lambda K: Kstep1(C1, grid, lr, F_b, F_b_top_bottom, DX, DY, 
                                        option_scheme=option_scheme, 
                                        DX_0=DX_0, DX_1=DX_1, DY_0=DY_0, DY_1=DY_1), dt)
    elif option_timescheme == "impl_Euler" or option_timescheme == "impl_Euler_gmres":
        K = impl_Euler(K, lambda K: Kstep1(C1, grid, lr, F_b, F_b_top_bottom, DX, DY, 
                                        option_scheme=option_scheme, 
                                        DX_0=DX_0, DX_1=DX_1, DY_0=DY_0, DY_1=DY_1), dt,
                                        option = option_timescheme)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= (np.sqrt(grid.dx) * np.sqrt(grid.dy))
    lr.S *= (np.sqrt(grid.dx) * np.sqrt(grid.dy))

    # S step
    D1 = computeD(
        lr, grid, F_b, F_b_top_bottom, DX=DX, DY=DY, dimensions="2x1d", option_dd="dd"
    )
    if option_timescheme == "RK4":
        lr.S += dt * RK4(lr.S, lambda S: Sstep1(C1, D1, grid), dt)
    elif option_timescheme == "impl_Euler" or option_timescheme == "impl_Euler_gmres":
        lr.S = impl_Euler(lr.S, lambda S: Sstep1(C1, D1, grid), dt,
                          option = option_timescheme)

    # L step
    L = lr.V @ lr.S.T
    B1 = computeB(L, grid, dimensions="2x1d")
    if option_timescheme == "RK4":
        L += dt * RK4(L, lambda L: Lstep1(lr, D1, grid), dt)
    elif option_timescheme == "impl_Euler" or option_timescheme == "impl_Euler_gmres":
        L = impl_Euler(L, lambda L: Lstep1(lr, D1, grid), dt,
                          option = option_timescheme)
    lr.V, St = np.linalg.qr(L, mode="reduced")
    lr.S = St.T
    lr.V /= np.sqrt(grid.dphi)
    lr.S *= np.sqrt(grid.dphi)

    ### Drop basis for adaptive rank strategy:
    lr, grid = drop_basis_functions(lr, grid, drop_tol, dimensions="2x1d")
    
    if rank_dropped is not None:
        rank_dropped.append(grid.r)

    # Step 2: advection in y

    ### Add basis for adaptive rank strategy:
    if option_rank_adaptivity == "v1":
        lr, grid = add_basis_functions(
            lr, grid, F_b_top_bottom, tol_sing_val, dimensions="2x1d"
        )
    else:
        lr, grid = add_basis_functions_v2(
            lr, grid, F_b_top_bottom, drop_tol
        )

    # K step
    C1, C2 = computeC(lr, grid, dimensions="2x1d")
    K = lr.U @ lr.S
    if option_timescheme == "RK4":
        K += dt * RK4(K, lambda K: Kstep2(C1, grid, lr, F_b, F_b_top_bottom, DX, DY, 
                                        option_scheme=option_scheme, 
                                        DX_0=DX_0, DX_1=DX_1, DY_0=DY_0, DY_1=DY_1), dt)
    elif option_timescheme == "impl_Euler" or option_timescheme == "impl_Euler_gmres":
        K = impl_Euler(K, lambda K: Kstep2(C1, grid, lr, F_b, F_b_top_bottom, DX, DY, 
                                        option_scheme=option_scheme, 
                                        DX_0=DX_0, DX_1=DX_1, DY_0=DY_0, DY_1=DY_1), dt,
                          option = option_timescheme)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= (np.sqrt(grid.dx) * np.sqrt(grid.dy))
    lr.S *= (np.sqrt(grid.dx) * np.sqrt(grid.dy))

    # S step
    D1 = computeD(
        lr, grid, F_b, F_b_top_bottom, DX=DX, DY=DY, dimensions="2x1d", option_dd="dd"
    )
    if option_timescheme == "RK4":
        lr.S += dt * RK4(lr.S, lambda S: Sstep2(C1, D1, grid), dt)
    elif option_timescheme == "impl_Euler" or option_timescheme == "impl_Euler_gmres":
        lr.S = impl_Euler(lr.S, lambda S: Sstep2(C1, D1, grid), dt,
                          option = option_timescheme)

    # L step
    L = lr.V @ lr.S.T
    if option_timescheme == "RK4":
        L += dt * RK4(L, lambda L: Lstep2(lr, D1, grid), dt)
    elif option_timescheme == "impl_Euler" or option_timescheme == "impl_Euler_gmres":
        L = impl_Euler(L, lambda L: Lstep2(lr, D1, grid), dt,
                          option = option_timescheme)
    lr.V, St = np.linalg.qr(L, mode="reduced")
    lr.S = St.T
    lr.V /= np.sqrt(grid.dphi)
    lr.S *= np.sqrt(grid.dphi)

    ### Drop basis for adaptive rank strategy:
    lr, grid = drop_basis_functions(lr, grid, drop_tol, dimensions="2x1d")

    # Step 3: collisions and source

    # K step
    C1, C2 = computeC(lr, grid, dimensions="2x1d")
    K = lr.U @ lr.S
    if option_timescheme == "RK4":
        K += dt * RK4(K, lambda K: Kstep3(K, C2, grid, lr, source=source), dt)
    elif option_timescheme == "impl_Euler" or option_timescheme == "impl_Euler_gmres":
        K = impl_Euler(K, lambda K: Kstep3(K, C2, grid, lr, source=source), dt,
                          option = option_timescheme)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= (np.sqrt(grid.dx) * np.sqrt(grid.dy))
    lr.S *= (np.sqrt(grid.dx) * np.sqrt(grid.dy))

    # S step
    if option_timescheme == "RK4":
        lr.S += dt * RK4(lr.S, lambda S: Sstep3(S, C2, grid, lr, source=source), dt)
    elif option_timescheme == "impl_Euler" or option_timescheme == "impl_Euler_gmres":
        lr.S = impl_Euler(lr.S, lambda S: Sstep3(S, C2, grid, lr, source=source), dt,
                          option = option_timescheme)

    # L step
    L = lr.V @ lr.S.T
    B1 = computeB(L, grid, dimensions="2x1d")
    if option_timescheme == "RK4":
        L += dt * RK4(L, lambda L: Lstep3(L, B1, grid, lr, source=source), dt)
    elif option_timescheme == "impl_Euler" or option_timescheme == "impl_Euler_gmres":
        L = impl_Euler(L, lambda L: Lstep3(L, B1, grid, lr, source=source), dt,
                          option = option_timescheme)
    lr.V, St = np.linalg.qr(L, mode="reduced")
    lr.S = St.T
    lr.V /= np.sqrt(grid.dphi)
    lr.S *= np.sqrt(grid.dphi)

    return lr, grid, rank_adapted, rank_dropped


def PSI_splitting_strang(
    lr,
    grid,
    dt,
    F_b,
    F_b_top_bottom,
    DX=None,
    DY=None,
    lr_periodic=None,
    location="left",
    tol_sing_val=1e-6,
    drop_tol=1e-6,
    rank_adapted=None,
    rank_dropped=None,
):
    """
    Old version of Projector splitting integrator with equation splitting and 
    strang splitting.

    Not used anymore, only kept for further development.
    """

    # ToDo: Still need to add domain decomposition in Y to strang splitting, 
    # for now it does not work at all.

    # Step 1: advection in x

    ### Add basis for adaptive rank strategy:
    lr, grid = add_basis_functions(
        lr, grid, F_b, tol_sing_val, dimensions="2x1d"
    )

    # 1/2 K step
    C1, C2 = computeC(lr, grid, dimensions="2x1d")
    K = lr.U @ lr.S
    K += 0.5 * dt * RK4(K, lambda K: Kstep1(C1, grid, lr, F_b, DX, DY), 0.5 * dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= (np.sqrt(grid.dx) * np.sqrt(grid.dy))
    lr.S *= (np.sqrt(grid.dx) * np.sqrt(grid.dy))

    # 1/2 S step
    D1 = computeD(
        lr, grid, F_b, F_b_top_bottom, DX=DX, DY=DY, dimensions="2x1d", option_dd="dd"
    )
    lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep1(C1, D1, grid), 0.5 * dt)

    # L step
    L = lr.V @ lr.S.T
    B1 = computeB(L, grid, dimensions="2x1d")
    L += dt * RK4(L, lambda L: Lstep1(lr, D1, grid), dt)
    lr.V, St = np.linalg.qr(L, mode="reduced")
    lr.S = St.T
    lr.V /= np.sqrt(grid.dphi)
    lr.S *= np.sqrt(grid.dphi)

    # Compute F_b
    if location == "left":
        F_b = computeF_b_2x1d_X(
            lr.U @ lr.S @ lr.V.T,
            grid,
            f_right=lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T,
            f_periodic=lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T,
        )  # recalculate F_b at time t + 0.5 dt
    elif location == "right":
        F_b = computeF_b_2x1d_X(
            lr.U @ lr.S @ lr.V.T,
            grid,
            f_left=lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T,
            f_periodic=lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T,
        )  # recalculate F_b at time t + 0.5 dt

    D1 = computeD(
        lr, grid, F_b, F_b_top_bottom, DX=DX, DY=DY, dimensions="2x1d", option_dd="dd"
    )  # recalculate D1 because we recalculated F_b

    # 1/2 S step
    C1, C2 = computeC(
        lr, grid, dimensions="2x1d"
    )  # need to recalculate C1 and C2 because we changed V in L step
    lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep1(C1, D1, grid), 0.5 * dt)

    # 1/2 K step
    K = lr.U @ lr.S
    K += 0.5 * dt * RK4(K, lambda K: Kstep1(C1, grid, lr, F_b, DX, DY), 0.5 * dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= (np.sqrt(grid.dx) * np.sqrt(grid.dy))
    lr.S *= (np.sqrt(grid.dx) * np.sqrt(grid.dy))

    ### Drop basis for adaptive rank strategy:
    lr, grid = drop_basis_functions(lr, grid, drop_tol, dimensions="2x1d")

    # Step 2: advection in y

    ### Add basis for adaptive rank strategy:
    lr, grid = add_basis_functions(
        lr, grid, F_b_top_bottom, tol_sing_val, dimensions="2x1d"
    )
    if rank_adapted is not None:
        rank_adapted.append(grid.r)

    # 1/2 K step
    C1, C2 = computeC(lr, grid, dimensions="2x1d")
    K = lr.U @ lr.S
    K += 0.5 * dt * RK4(K, lambda K: Kstep2(K, C1, grid, DX, DY), 0.5 * dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= (np.sqrt(grid.dx) * np.sqrt(grid.dy))
    lr.S *= (np.sqrt(grid.dx) * np.sqrt(grid.dy))

    # 1/2 S step
    D1 = computeD(
        lr, grid, F_b, F_b_top_bottom, DX=DX, DY=DY, dimensions="2x1d", option_dd="dd"
    )
    lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep2(S, C1, D1, grid), 0.5 * dt)

    # L step
    L = lr.V @ lr.S.T
    L += dt * RK4(L, lambda L: Lstep2(L, D1, grid), dt)
    lr.V, St = np.linalg.qr(L, mode="reduced")
    lr.S = St.T
    lr.V /= np.sqrt(grid.dphi)
    lr.S *= np.sqrt(grid.dphi)

    # Compute F_b
    if location == "left":
        F_b = computeF_b_2x1d_X(
            lr.U @ lr.S @ lr.V.T,
            grid,
            f_right=lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T,
            f_periodic=lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T,
        )  # recalculate F_b at time t + 0.5 dt
    elif location == "right":
        F_b = computeF_b_2x1d_X(
            lr.U @ lr.S @ lr.V.T,
            grid,
            f_left=lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T,
            f_periodic=lr_periodic.U @ lr_periodic.S @ lr_periodic.V.T,
        )  # recalculate F_b at time t + 0.5 dt

    D1 = computeD(
        lr, grid, F_b, F_b_top_bottom, DX=DX, DY=DY, dimensions="2x1d", option_dd="dd"
    )  # maybe we dont even have to recompute here

    # 1/2 S step
    C1, C2 = computeC(
        lr, grid, dimensions="2x1d"
    )  # need to recalculate C1 and C2 because we changed V in L step
    lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep2(S, C1, D1, grid), 0.5 * dt)

    # 1/2 K step
    K = lr.U @ lr.S
    K += 0.5 * dt * RK4(K, lambda K: Kstep2(K, C1, grid, DX, DY), 0.5 * dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= (np.sqrt(grid.dx) * np.sqrt(grid.dy))
    lr.S *= (np.sqrt(grid.dx) * np.sqrt(grid.dy))

    ### Drop basis for adaptive rank strategy:
    lr, grid = drop_basis_functions(lr, grid, drop_tol, dimensions="2x1d")
    if rank_dropped is not None:
        rank_dropped.append(grid.r)

    # Step 3: collisions

    # 1/2 K step
    C1, C2 = computeC(lr, grid, dimensions="2x1d")
    K = lr.U @ lr.S
    K += 0.5 * dt * RK4(K, lambda K: Kstep3(K, C2, grid), 0.5 * dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= (np.sqrt(grid.dx) * np.sqrt(grid.dy))
    lr.S *= (np.sqrt(grid.dx) * np.sqrt(grid.dy))

    # 1/2 S step
    lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep3(S, C2, grid), 0.5 * dt)

    # L step
    L = lr.V @ lr.S.T
    B1 = computeB(L, grid, dimensions="2x1d")
    L += dt * RK4(L, lambda L: Lstep3(L, B1, grid), dt)
    lr.V, St = np.linalg.qr(L, mode="reduced")
    lr.S = St.T
    lr.V /= np.sqrt(grid.dphi)
    lr.S *= np.sqrt(grid.dphi)

    # 1/2 S step
    C1, C2 = computeC(
        lr, grid, dimensions="2x1d"
    )  # need to recalculate C1 and C2 because we changed V in L step
    lr.S += 0.5 * dt * RK4(lr.S, lambda S: Sstep3(S, C2, grid), 0.5 * dt)

    # 1/2 K step
    K = lr.U @ lr.S
    K += 0.5 * dt * RK4(K, lambda K: Kstep3(K, C2, grid), 0.5 * dt)
    lr.U, lr.S = np.linalg.qr(K, mode="reduced")
    lr.U /= (np.sqrt(grid.dx) * np.sqrt(grid.dy))
    lr.S *= (np.sqrt(grid.dx) * np.sqrt(grid.dy))

    return lr, grid, rank_adapted, rank_dropped
