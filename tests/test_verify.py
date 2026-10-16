"""verify(): global first, then a local region; and soundness by simulation."""

import numpy as np
import pytest
import nnsector as ns

import nnstab as nst


def test_global_example_is_verified_globally():
    A, B, C, net = nst.example_system("global")
    r = nst.verify(A, B, C, net)
    assert r.status == "global" and r.stable and r.local_check is None
    assert r.contains([5.0, 1e3, 0.0]) and not r.contains([-1.0, 0, 0])
    assert "Globally exponentially stable" in str(r)


def test_local_example_finds_the_largest_box():
    A, B, C, net = nst.example_system("local")
    r = nst.verify(A, B, C, net)
    c = r.local_check
    assert r.status == "local" and c.certified and r.boxes_tried > 5
    assert c.ybar.item() == pytest.approx(1.848, abs=2e-3)
    assert nst.check_local(A, B, C, net, c.ybar * 1.01).hurwitz is False      # just past the edge
    assert "min_decay" in r.note and "Locally exponentially stable" in str(r)


def test_min_decay_gives_a_smaller_box_with_a_faster_rate():
    A, B, C, net = nst.example_system("local")
    r = nst.verify(A, B, C, net, min_decay=0.5)
    assert r.status == "local" and r.local_check.decay >= 0.5
    assert r.local_check.ybar.item() == pytest.approx(1.440, abs=2e-3)
    assert r.note == ""


def test_given_box():
    A, B, C, net = nst.example_system("local")
    assert nst.verify(A, B, C, net, ybar=1.0).status == "local"
    r = nst.verify(A, B, C, net, ybar=3.0)
    assert r.status == "none" and not r.stable and not r.contains(np.zeros(3))
    assert "does not mean" in str(r)


def test_not_verified_when_even_small_boxes_fail():
    A, B, C, net = nst.example_system("local")
    r = nst.verify(A + 5 * np.eye(3), B, C, net)          # far too unstable for this controller
    assert r.status == "none" and "very small boxes" in r.note
    assert not r.local_check.hurwitz


def test_search_notes_when_the_local_test_passes_on_huge_boxes():
    # pi(y) = -tanh(y) + 0.5 tanh(y) = -0.5 tanh(y), but the global bound is 1.5, too loose
    # for the global test; the local test passes on every box
    A = np.array([[-2.0, 1.0], [2.0, -2.0]])
    B, C = np.array([[1.0], [0.0]]), np.array([[1.0, 0.0]])
    net = ns.Network([[[1.0], [1.0]], [[-1.0, 0.5]]])
    r = nst.verify(A, B, C, net)
    assert not r.global_check.certified and not r.global_check.hurwitz
    assert r.status == "local" and "very large boxes" in r.note


def test_two_inputs_and_a_direction():
    # an unstable plant; pi_k(y) = -tanh(2 y_k) - 0.5 tanh(y_k + 0.3 y_other)
    A = np.array([[-3.0, 2.5, 2.0], [2.0, -3.0, 2.5], [2.5, 2.0, -4.0]])
    B = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    C = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])
    net = ns.Network([[[2.0, 0.0], [0.0, 2.0], [1.0, 0.3], [0.3, 1.0]],
                      [[-1.0, 0.0, -0.5, 0.0], [0.0, -1.0, 0.0, -0.5]]])
    boxes = {None: [1.8564453125] * 2, (1.0, 2.0): [1.3359375, 2.671875]}
    for d, box in boxes.items():
        r = nst.verify(A, B, C, net, direction=d)
        c = r.local_check
        assert r.status == "local" and not r.global_check.hurwitz
        assert c.gamma1.shape == (2, 2) and np.allclose(c.ybar, box, rtol=1e-12, atol=0)
        corners = c.rho / c.v * np.eye(3)
        assert np.all(corners @ C.T <= c.ybar * (1 + 1e-12))
        assert np.max(corners @ C.T / c.ybar) == pytest.approx(1.0)


def soundness_cases():
    rng = np.random.default_rng(11)
    for t in range(25):
        n = int(rng.integers(2, 5))
        A = np.abs(rng.standard_normal((n, n))) * 1.5 - rng.uniform(2.0, 4.0) * np.eye(n)
        B = np.abs(rng.standard_normal((n, 1))) * (rng.random((n, 1)) < 0.8)
        C = np.abs(rng.standard_normal((1, n))) * (rng.random((1, n)) < 0.8)
        h = int(rng.integers(3, 9))
        act = ["tanh", "relu", "leakyrelu"][t % 3]
        W = [rng.standard_normal((h, 1)), rng.standard_normal((h, h)) / np.sqrt(h),
             -np.abs(rng.standard_normal((1, h))) / np.sqrt(h)]
        yield A, B, C, ns.Network(W, activation=act)


@pytest.mark.parametrize("case", list(soundness_cases()))
def test_certificates_hold_in_simulation(case):
    A, B, C, net = case
    r = nst.verify(A, B, C, net)
    if r.status == "local":
        c = r.local_check
        X0 = np.vstack([nst.sample_region(c, 40, seed=0), c.rho / c.v * np.eye(len(A))])
        t, X = nst.simulate(A, B, C, net, X0, t_end=10.0, dt=0.01)
        V = X @ c.v
        assert X.min() >= -1e-9                                       # stays nonnegative
        assert np.all(V <= c.rho * (1 + 1e-7))                        # stays in the region
        assert np.all(V <= V[0] * np.exp(-c.decay * t)[:, None] * (1 + 1e-6) + 1e-12)
    elif r.status == "global":
        g = r.global_check
        X0 = np.random.default_rng(0).exponential(scale=10.0, size=(40, len(A)))
        t, X = nst.simulate(A, B, C, net, X0, t_end=10.0, dt=0.01)
        S = X.sum(axis=2)
        assert X.min() >= -1e-9
        assert np.all(S <= g.overshoot * np.exp(-g.decay * t)[:, None] * S[0] * (1 + 1e-6) + 1e-12)
