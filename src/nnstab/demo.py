"""A small controller and two plants shipped with the package, for examples and tests."""

from __future__ import annotations

from importlib import resources

import numpy as np
import nnsector as ns


def example_controller() -> ns.Network:
    """A bias-free tanh network trained to act like the controller u = -tanh(2y) - 0.5y.

    One input, two hidden layers of 8 tanh neurons, one output, no biases,
    so pi(0) = 0. It stays within 0.006 of the controller for 0 <= y <= 3.
    Near y = 0, pi(y) is about -2.5 y. For large y its output levels off
    (it was trained only for |y| <= 3), so its gain |pi(y) / y| goes to 0.
    """
    data = resources.files("nnstab") / "data"
    weights = []
    for i in (1, 2, 3):
        with (data / f"controller_W{i}.csv").open("r") as fh:
            weights.append(np.loadtxt(fh, delimiter=",", ndmin=2))
    return ns.Network(weights, activation="tanh")


def example_system(kind: str = "local"):
    """A positive plant with :func:`example_controller` in the loop; returns (A, B, C, net).

    'local': an open-loop unstable plant. The controller stabilizes it near
    the origin, where its gain is high, but not for large outputs, where its
    gain is low. The global test fails and a local region is found.

    'global': a stable plant that the controller only nudges. The global
    test passes.
    """
    net = example_controller()
    if kind == "local":
        A = np.array([[-3.0, 3.5, 2.0],
                      [2.0, -4.0, 1.0],
                      [3.5, 3.5, -5.0]])
        B = np.array([[1.0], [0.5], [1.0]])
        C = np.array([[1.0, 1.0, 0.5]])
    elif kind == "global":
        A = np.array([[-2.0, 1.0, 0.5],
                      [0.5, -1.5, 0.5],
                      [1.0, 0.5, -2.5]])
        B = np.array([[0.1], [0.0], [0.2]])
        C = np.array([[0.0, 0.3, 0.1]])
    else:
        raise ValueError("kind must be 'local' or 'global'.")
    return A, B, C, net
