"""Tests for DLR_rt/src/d1x1/lowrank.py, the 1x1d low-rank step on its own."""
import numpy as np
import pytest

from DLR_rt.src.d1x1.lowrank import (
    LR,
    add_basis,
    boundary_values,
    drop_basis,
    psi_step,
)
from DLR_rt.src.grid import Grid_1x1d
from DLR_rt.src.initial_condition import setInitialCondition_1x1d_lr
from DLR_rt.tests.make_reference import DT, NMU, NX, R, SEED, T_F, TOL
from DLR_rt.tests.make_reference import PATH as REFERENCE


def initial(n_domains: int, Nx=NX, Nmu=NMU, r=R):
    """Subgrids, initial LRs and one generator per subdomain, as in setup_domains."""
    np.random.seed(SEED)
    grids = Grid_1x1d(Nx, Nmu, r).split_n(n_domains)
    lrs = []
    for g in grids:
        lr = setInitialCondition_1x1d_lr(g)
        lrs.append(LR(lr.U, lr.S, lr.V))
    rngs = [np.random.default_rng([SEED, i]) for i in range(n_domains)]
    return grids, lrs, rngs


def run(n_domains: int):
    """Plain serial loop with tanh(t) inflow on both physical boundaries."""
    grids, lrs, rngs = initial(n_domains)
    n = len(lrs)
    rank_hist = [[lr.r] for lr in lrs]
    t, dt = 0.0, DT
    while t < T_F:
        dt = min(dt, T_F - t)
        bc = np.full(NMU, np.tanh(t))
        F_bs = [boundary_values(lrs[i], grids[i],
                                lrs[i - 1].last_row() if i > 0 else bc,
                                lrs[i + 1].first_row() if i < n - 1 else bc)
                for i in range(n)]
        for i in range(n):
            lr = add_basis(lrs[i], grids[i], F_bs[i], TOL, rngs[i])
            lr = psi_step(lr, grids[i], F_bs[i], dt)
            lrs[i] = drop_basis(lr, grids[i], TOL)
            rank_hist[i].append(lrs[i].r)
        t += dt
    return lrs, rank_hist


@pytest.mark.parametrize("n_domains", [1, 2])
def test_matches_reference(n_domains):
    """Same result as the old dlr_inflow (1 domain) and dlr_dd (2 domains)."""
    ref = np.load(REFERENCE)
    lrs, rank_hist = run(n_domains)
    f = np.concatenate([lr.full() for lr in lrs], axis=0)
    # Exact in the same environment, allclose so another BLAS does not fail
    np.testing.assert_allclose(f, ref[f"f_{n_domains}dom"], rtol=0, atol=1e-12)
    assert rank_hist == ref[f"rank_hist_{n_domains}dom"].tolist()


def test_edge_rows_and_boundary_values():
    """first/last_row are the rows of full(), F_b: inflow in, extrapolation out."""
    grid, lr, _ = (x[0] for x in initial(1))
    f = lr.full()
    assert np.allclose(lr.first_row(), f[0], rtol=0, atol=1e-13)
    assert np.allclose(lr.last_row(), f[-1], rtol=0, atol=1e-13)

    left, right = np.full(NMU, 1.5), np.full(NMU, -2.0)
    F_b = boundary_values(lr, grid, left, right)
    pos, neg = grid.MU > 0, grid.MU < 0
    assert np.array_equal(F_b[0, pos], left[pos])
    assert np.array_equal(F_b[1, neg], right[neg])
    np.testing.assert_allclose(F_b[0, neg], (2 * f[0] - f[1])[neg], atol=1e-13)
    np.testing.assert_allclose(F_b[1, pos], (2 * f[-1] - f[-2])[pos], atol=1e-13)


def test_add_basis_keeps_f_and_orthonormality():
    """Adding basis functions changes the rank, but not f."""
    grid, lr, rng = (x[0] for x in initial(1))
    F_b = boundary_values(lr, grid, np.ones(NMU), np.ones(NMU))
    new = add_basis(lr, grid, F_b, TOL, rng)
    assert new.r > lr.r
    np.testing.assert_allclose(new.full(), lr.full(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(new.U.T @ new.U * grid.dx, np.eye(new.r), atol=1e-12)
    np.testing.assert_allclose(new.V.T @ new.V * grid.dmu, np.eye(new.r), atol=1e-12)


def test_drop_basis_min_rank_and_truncation():
    """A rank 2 S on r = 8 is cut to min_rank, the error is at round-off."""
    grid, lr, _ = (x[0] for x in initial(1, r=8))
    S = np.zeros((8, 8))
    S[0, 0], S[1, 1] = 1.0, 0.5
    lr = LR(lr.U, S, lr.V)
    assert drop_basis(lr, grid, TOL, min_rank=5).r == 5
    small = drop_basis(lr, grid, TOL, min_rank=1)
    assert small.r == 2
    np.testing.assert_allclose(small.full(), lr.full(), rtol=0, atol=1e-12)


def test_psi_step_constant_state():
    """
    f = c(t) uniform, inflow c(t) on both sides: no transport, only scattering.

    MU includes both endpoints, so sum(dmu) = 2 Nmu / (Nmu - 1) instead of 2 and
    the discrete scattering 1/2 int f dmu - f is c / (Nmu - 1), not 0.
    f has to stay uniform and grow as exp(t / (Nmu - 1)).
    """
    grid = Grid_1x1d(64, NMU, 1)
    lr = LR(np.ones((grid.Nx, 1)) / np.sqrt(grid.Nx * grid.dx),
            np.array([[np.sqrt(grid.Nx * grid.dx * grid.Nmu * grid.dmu)]]),
            np.ones((grid.Nmu, 1)) / np.sqrt(grid.Nmu * grid.dmu))
    dt, steps = 1e-3, 10
    for _ in range(steps):
        c = lr.full()[0, 0]
        lr = psi_step(lr, grid, boundary_values(lr, grid, np.full(NMU, c),
                                                np.full(NMU, c)), dt)
    f = lr.full()
    np.testing.assert_allclose(f, f[0, 0], rtol=1e-13, atol=0)
    # growth is 3e-4, the splitting error 2e-7
    np.testing.assert_allclose(f[0, 0], np.exp(steps * dt / (NMU - 1)), rtol=1e-6)
