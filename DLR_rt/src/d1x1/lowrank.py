"""
Low-rank representation and one projector-splitting (PSI) step, 1x1d inflow only.

f(x, mu) = U S V^T with the weighted orthonormality U^T U dx = I, V^T V dmu = I.
One time step on a subdomain is

    F_b = boundary_values(lr, grid, inflow_left, inflow_right)
    lr = add_basis(lr, grid, F_b, tol, rng)
    lr = psi_step(lr, grid, F_b, dt)            # Lie splitting K -> S -> L, RK4
    lr = drop_basis(lr, grid, tol)

The grid (Grid_1x1d) is only read: Nx, Nmu, dx, dmu, MU, coeff. The rank is lr.r.
MU has to be sorted with Nmu even, mu < 0 in the first half and mu > 0 in the second.

Bit-identical to the old general code (lr.py / integrators.PSI_lie with
option_scheme="upwind"), including two properties of that code:
  - the transport terms (dxK in the K step, D1 in the S and L step) are computed once
    from the state at the start of the sub-step and stay fixed in the RK4 stages,
  - the K step uses upwind differences, D1 for the S and L step centered differences.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class LR:
    """f = U S V^T, U: (Nx, r), S: (r, r), V: (Nmu, r)."""
    U: np.ndarray
    S: np.ndarray
    V: np.ndarray

    @property
    def r(self) -> int:
        return self.S.shape[0]

    def full(self) -> np.ndarray:
        """The full matrix f, shape (Nx, Nmu)."""
        return self.U @ self.S @ self.V.T

    def first_row(self) -> np.ndarray:
        """f at the first x grid point, is sent to the left neighbour."""
        return self.U[0, :] @ self.S @ self.V.T

    def last_row(self) -> np.ndarray:
        """f at the last x grid point, is sent to the right neighbour."""
        return self.U[-1, :] @ self.S @ self.V.T


### Boundary

def boundary_values(lr: LR, grid, inflow_left: np.ndarray,
                    inflow_right: np.ndarray) -> np.ndarray:
    """
    Values of f just outside the subdomain, F_b of shape (2, Nmu).

    F_b[0] left, F_b[1] right. Incoming directions (mu > 0 left, mu < 0 right) take
    the inflow, i.e. the edge row of the neighbour or the physical boundary condition.
    Outgoing directions are extrapolated linearly from the inside, mu = 0 stays 0.
    """
    rows = lr.U[[0, 1, -2, -1], :] @ lr.S @ lr.V.T  # only the 4 rows needed, (4, Nmu)
    pos = grid.MU > 0
    neg = grid.MU < 0

    F_b = np.zeros((2, grid.Nmu))
    F_b[0, pos] = inflow_left[pos]
    F_b[0, neg] = 2 * rows[0, neg] - rows[1, neg]
    F_b[1, pos] = 2 * rows[3, pos] - rows[2, pos]
    F_b[1, neg] = inflow_right[neg]
    return F_b


def _k_boundary(lr: LR, grid, F_b: np.ndarray):
    """
    Boundary values of K = U S projected onto V, the ghost points of dxK.

    Incoming half: the inflow F_b. Outgoing half: K extrapolated linearly.
    """
    half = grid.Nmu // 2  # mu < 0 | mu > 0
    V_neg, V_pos = lr.V[:half], lr.V[half:]

    K = lr.U @ lr.S
    K_extra_left = 2 * K[0] - K[1]
    K_extra_right = 2 * K[-1] - K[-2]

    K_left = (F_b[0, half:] @ V_pos) * grid.dmu \
        + (K_extra_left @ (V_neg.T @ V_neg) * grid.dmu)
    K_right = (F_b[1, :half] @ V_neg) * grid.dmu \
        + (K_extra_right @ (V_pos.T @ V_pos) * grid.dmu)
    return K_left, K_right


### Derivatives in x, O(Nx r) without difference matrices

def _dx_centered(K: np.ndarray, K_left: np.ndarray, K_right: np.ndarray,
                 dx: float) -> np.ndarray:
    """(K[i+1] - K[i-1]) / 2dx, with the ghost points K_left, K_right."""
    dxK = np.empty_like(K)
    dxK[1:-1] = K[2:] - K[:-2]
    dxK[0] = K[1]
    dxK[-1] = -K[-2]
    dxK /= 2 * dx
    dxK[0, :] -= K_left / (2 * dx)
    dxK[-1, :] += K_right / (2 * dx)
    return dxK


def _dx_upwind(KP: np.ndarray, KP_left: np.ndarray, KP_right: np.ndarray,
               eigvals: np.ndarray, dx: float) -> np.ndarray:
    """
    Upwind derivative of K P (columns in the eigenbasis of C1).

    Column j is transported with speed eigvals[j]: backward difference if > 0,
    forward difference otherwise.
    """
    c = 1 / dx
    backward = c * KP  # (KP[i] - KP[i-1]) / dx
    backward[1:] -= c * KP[:-1]
    backward[0, :] -= KP_left / dx
    forward = -c * KP  # (KP[i+1] - KP[i]) / dx
    forward[:-1] += c * KP[1:]
    forward[-1, :] += KP_right / dx
    return np.where(eigvals > 0, backward, forward)


### Projector splitting integrator

def _rk4(y: np.ndarray, rhs, dt: float) -> np.ndarray:
    """Averaged slope of one classical RK4 step, y_new = y + dt * _rk4(y, rhs, dt)."""
    k0 = rhs(y)
    k1 = rhs(y + dt * 0.5 * k0)
    k2 = rhs(y + dt * 0.5 * k1)
    k3 = rhs(y + dt * k2)
    return 1.0 / 6.0 * k0 + 1.0 / 3.0 * k1 + 1.0 / 3.0 * k2 + 1.0 / 6.0 * k3


def coefficients(V: np.ndarray, grid):
    """C1 = int mu V V^T dmu, shape (r, r) and C2 = int V dmu, shape (1, r)."""
    C1 = ((V.T * grid.MU) @ V) * grid.dmu
    C2 = (V.T @ np.ones((grid.Nmu, 1))).T * grid.dmu
    return C1, C2


def psi_step(lr: LR, grid, F_b: np.ndarray, dt: float) -> LR:
    """
    One Lie splitting step K -> S -> L of
        df/dt = -coeff mu df/dx + coeff^2 (1/2 int f dmu - f)
    with the inflow F_b. Returns a new LR, the rank stays the same.
    """
    c = grid.coeff
    U, S, V = lr.U, lr.S, lr.V
    C1, C2 = coefficients(V, grid)

    # K step: K = U S, V fixed. Upwind in the eigenbasis of C1 = P diag(eigvals) P^T
    K = U @ S
    eigvals, P = np.linalg.eigh(C1)
    K_left, K_right = _k_boundary(lr, grid, F_b)
    dxKP = _dx_upwind(K @ P, K_left @ P, K_right @ P, eigvals, grid.dx)
    transport = ((-c * dxKP) * eigvals) @ P.T  # = -c dxK C1
    K = K + dt * _rk4(
        K, lambda K: transport + 0.5 * c**2 * K @ C2.T @ C2 - c**2 * K, dt)
    U, S = np.linalg.qr(K, mode="reduced")
    U /= np.sqrt(grid.dx)
    S *= np.sqrt(grid.dx)

    # S step: U, V fixed. D1 = int U^T dx(U S) dx, centered differences
    lr_k = LR(U, S, V)
    K_left, K_right = _k_boundary(lr_k, grid, F_b)
    D1 = U.T @ _dx_centered(U @ S, K_left, K_right, grid.dx) * grid.dx
    transport = c * D1 @ C1
    S = S + dt * _rk4(
        S, lambda S: transport - 0.5 * c**2 * S @ C2.T @ C2 + c**2 * S, dt)

    # L step: L = V S^T, U fixed
    L = V @ S.T
    B1 = (L.T @ np.ones((grid.Nmu, 1))).T * grid.dmu
    transport = ((-c * grid.MU)[:, None] * V) @ D1.T
    scatter = 0.5 * c**2 * np.ones((grid.Nmu, 1)) @ B1
    L = L + dt * _rk4(L, lambda L: transport + scatter - c**2 * L, dt)
    V, St = np.linalg.qr(L, mode="reduced")
    V /= np.sqrt(grid.dmu)
    S = St.T * np.sqrt(grid.dmu)

    return LR(U, S, V)


### Rank adaptivity

def add_basis(lr: LR, grid, F_b: np.ndarray, tol: float,
              rng: np.random.Generator) -> LR:
    """
    Extend the basis by the dominant directions of the inflow F_b.

    V gets the right singular vectors of F_b with singular value above tol (scaled),
    U as many random vectors from rng, S is padded with zeros, so f does not change.
    Give every subdomain its own rng, then the result does not depend on the order
    in which the subdomains are updated (serial vs. MPI).
    """
    X, sing_val, QT = np.linalg.svd(F_b, full_matrices=False)
    sing_val *= np.sqrt(grid.dx * grid.dmu)
    QT /= np.sqrt(grid.dmu)

    r_b = np.sum(sing_val > tol * np.sqrt(grid.Nx * grid.Nmu * (grid.dx * grid.dmu)))
    r_b = min(r_b, grid.Nmu - lr.r)  # the rank cannot exceed Nmu

    U = np.concatenate((lr.U, rng.random((grid.Nx, r_b))), axis=1)
    V = np.concatenate((lr.V, QT.T[:, :r_b]), axis=1)
    S = np.zeros((lr.r + r_b, lr.r + r_b))
    S[:lr.r, :lr.r] = lr.S

    U, R_U = np.linalg.qr(U, mode="reduced")
    U /= np.sqrt(grid.dx)
    R_U *= np.sqrt(grid.dx)
    V, R_V = np.linalg.qr(V, mode="reduced")
    V /= np.sqrt(grid.dmu)
    R_V *= np.sqrt(grid.dmu)
    return LR(U, R_U @ S @ R_V.T, V)


def drop_basis(lr: LR, grid, tol: float, min_rank: int = 5) -> LR:
    """
    Truncate to the smallest rank whose dropped singular values of S have a
    2-norm below tol (scaled), but keep at least min_rank.
    """
    W, sing_val, ZT = np.linalg.svd(lr.S, full_matrices=False)

    threshold = (tol * np.sqrt(grid.Nx * grid.Nmu * (grid.dx * grid.dmu)))**2
    r = len(sing_val)
    sum_drop = sing_val[-1]**2
    while sum_drop < threshold and r > 0:
        r -= 1
        sum_drop += sing_val[r - 1]**2
    r = max(r, min_rank)

    return LR(lr.U @ W[:, :r], np.diag(sing_val[:r]), lr.V @ ZT.T[:, :r])
