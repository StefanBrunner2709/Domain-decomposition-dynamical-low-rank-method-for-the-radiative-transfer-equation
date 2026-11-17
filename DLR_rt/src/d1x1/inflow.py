"""
Inflow on the physical boundaries x = 0 (left) and x = 1 (right).

Every inflow is a function inflow(t, mu) -> (left, right), both of shape (Nmu,).
Only the incoming directions are used (mu > 0 of left, mu < 0 of right), see
lowrank.boundary_values. All of them are 0 at t = 0, consistent with f(0) = 0.

    INFLOWS["beam"](t, grid.MU)
"""
import numpy as np


def tanh(t: float, mu: np.ndarray):
    """Default: tanh(t) in all directions on both sides."""
    left = np.full(len(mu), np.tanh(t))
    return left, left.copy()


def beam(t: float, mu: np.ndarray, mu_min: float = 0.8, ramp: float = 20.0):
    """
    Narrow beam from the left, the right side is dark.

    mu_min: the beam enters in mu > mu_min, closer to 1 is narrower.
    ramp: switched on as tanh(ramp t), larger is sharper.
    """
    left = np.where(mu > mu_min, np.tanh(ramp * t), 0.0)
    return left, np.zeros_like(left)


def sharp_beam(t: float, mu: np.ndarray):
    """Narrower beam that is switched on faster."""
    return beam(t, mu, mu_min=0.95, ramp=50.0)


def pulse(t: float, mu: np.ndarray, mu_min: float = 0.8, ramp: float = 20.0,
          t_off: float = 0.2):
    """
    Beam from the left that is switched on as tanh(ramp t) and off around t_off.
    """
    envelope = np.tanh(ramp * t) * 0.5 * (1.0 - np.tanh(ramp * (t - t_off)))
    left = np.where(mu > mu_min, envelope, 0.0)
    return left, np.zeros_like(left)


def two_beam(t: float, mu: np.ndarray, mu_min: float = 0.8, ramp: float = 20.0):
    """Narrow beams from both sides, mirror images of each other, meet in the middle."""
    left = np.where(mu > mu_min, np.tanh(ramp * t), 0.0)
    right = np.where(mu < -mu_min, np.tanh(ramp * t), 0.0)
    return left, right


INFLOWS = {"tanh": tanh, "beam": beam, "sharp_beam": sharp_beam, "pulse": pulse,
           "two_beam": two_beam}
