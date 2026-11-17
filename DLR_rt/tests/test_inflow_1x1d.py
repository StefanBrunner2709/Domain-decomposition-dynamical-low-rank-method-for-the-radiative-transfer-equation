"""Tests for DLR_rt/src/d1x1/inflow.py."""
import numpy as np
import pytest

from DLR_rt.src.d1x1.inflow import INFLOWS, beam, pulse, two_beam

MU = np.linspace(-1, 1, 33)


@pytest.mark.parametrize("name", INFLOWS)
def test_shapes(name):
    left, right = INFLOWS[name](0.5, MU)
    assert left.shape == right.shape == MU.shape


@pytest.mark.parametrize("name", INFLOWS)
def test_zero_at_t0(name):
    """Consistent with the initial condition f = 0."""
    left, right = INFLOWS[name](0.0, MU)
    np.testing.assert_allclose(left, 0.0, atol=1e-3)
    np.testing.assert_allclose(right, 0.0, atol=1e-3)


def test_tanh_matches_old_default():
    """Same values as the old None -> np.full(Nmu, np.tanh(t)) in computeF_b_edges."""
    left, right = INFLOWS["tanh"](0.7, MU)
    assert np.array_equal(left, np.full(len(MU), np.tanh(0.7)))
    assert np.array_equal(right, left)


def test_beam_only_cone():
    """Nonzero only for mu > mu_min, the right boundary is dark."""
    left, right = beam(1.0, MU, mu_min=0.8)
    assert np.all(left[MU <= 0.8] == 0.0)
    assert np.all(left[MU > 0.8] > 0.99)
    assert np.all(right == 0.0)


def test_sharp_beam_narrower():
    left, _ = INFLOWS["sharp_beam"](1.0, MU)
    assert np.count_nonzero(left) < np.count_nonzero(beam(1.0, MU)[0])


def test_two_beam_symmetric():
    """The right beam is the mirror image of the left beam."""
    left, right = two_beam(1.0, MU)
    np.testing.assert_array_equal(left, right[::-1])


def test_pulse_switches_on_and_off():
    assert pulse(0.0, MU)[0].max() == 0.0
    assert pulse(0.1, MU, t_off=0.2)[0].max() > 0.9
    assert pulse(1.0, MU, t_off=0.2)[0].max() < 1e-6
