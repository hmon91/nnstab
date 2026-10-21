"""Simulation helpers, to look at the closed loop and to test a certificate on samples."""

from __future__ import annotations

import numpy as np

from .checks import LocalCheck, check_system


def simulate(A, B, C, net, x0, t_end: float, dt: float = 0.01):
    """Simulate x' = A x + B pi(C x) with the classical Runge-Kutta method (RK4).

    ``x0`` has shape (n,) for one initial state or (N, n) for N of them.
    Returns the times, shape (k+1,), and the states, shape (k+1, n) or
    (k+1, N, n).
    """
    A, B, C, net = check_system(A, B, C, net)
    x = np.asarray(x0, dtype=float)
    single = x.ndim == 1
    x = np.atleast_2d(x)
    if x.shape[1] != A.shape[0]:
        raise ValueError(f"x0 must have {A.shape[0]} entries per state.")
    if not (t_end > 0 and dt > 0):
        raise ValueError("t_end and dt must be > 0.")

    def f(z):
        return z @ A.T + net(z @ C.T) @ B.T

    steps = int(np.ceil(t_end / dt))
    h = t_end / steps
    out = np.empty((steps + 1,) + x.shape)
    out[0] = x
    for k in range(steps):
        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3 = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[k + 1] = x
    t = np.linspace(0.0, t_end, steps + 1)
    return (t, out[:, 0, :]) if single else (t, out)


def sample_region(check: LocalCheck, k: int, seed: int = 0) -> np.ndarray:
    """``k`` initial states drawn uniformly from the region {x >= 0 : v^T x <= rho}.

    Returns an array of shape (k, n).
    """
    if not check.certified:
        raise ValueError("The local check is not certified, so there is no region to sample.")
    rng = np.random.default_rng(seed)
    n = len(check.v)
    e = rng.exponential(size=(k, n + 1))
    u = e / e.sum(axis=1, keepdims=True)              # uniform on the simplex (with a slack entry)
    return check.rho * u[:, :n] / check.v[None, :]
