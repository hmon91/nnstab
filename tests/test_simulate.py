"""Simulation helpers."""

import numpy as np
import pytest
import nnsector as ns

import nnstab as nst


def test_rk4_on_a_linear_loop():
    # pi(y) = k y with no hidden layer: x' = (a + k) x
    A, B, C = np.array([[-1.0]]), np.array([[1.0]]), np.array([[1.0]])
    net = ns.Network([[[-0.5]]])
    t, X = nst.simulate(A, B, C, net, [2.0], t_end=3.0, dt=0.01)
    assert X.shape == (301, 1) and t[-1] == pytest.approx(3.0)
    assert np.allclose(X[:, 0], 2.0 * np.exp(-1.5 * t), rtol=1e-9)


def test_many_initial_states():
    A, B, C, net = nst.example_system("local")
    t, X = nst.simulate(A, B, C, net, np.ones((5, 3)) * 0.1, t_end=1.0, dt=0.1)
    assert X.shape == (11, 5, 3)
    with pytest.raises(ValueError, match="entries per state"):
        nst.simulate(A, B, C, net, np.ones(2), t_end=1.0)


def test_sample_region():
    A, B, C, net = nst.example_system("local")
    c = nst.check_local(A, B, C, net, 1.0)
    X = nst.sample_region(c, 500, seed=4)
    assert X.shape == (500, 3) and c.contains(X)
    assert np.max(X @ c.v) > 0.9 * c.rho                       # it reaches near the edge
    with pytest.raises(ValueError, match="not certified"):
        nst.sample_region(nst.check_local(A, B, C, net, 3.0), 5)
