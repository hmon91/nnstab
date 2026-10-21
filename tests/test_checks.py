"""The global and local checks, and the checks of the inputs."""

import numpy as np
import pytest
import nnsector as ns

import nnstab as nst


def test_global_check_on_the_global_example():
    A, B, C, net = nst.example_system("global")
    g = nst.check_global(A, B, C, net)
    assert g.certified and g.metzler and g.hurwitz and g.reason == ""
    assert g.Gamma2.item() == pytest.approx(4.79136, abs=1e-5)
    assert np.array_equal(g.Gamma1, -g.Gamma2)
    assert g.decay == pytest.approx(-nst.spectral_abscissa(g.M2), rel=1e-9)
    assert np.all(g.v @ g.M2 <= -g.decay * g.v * (1 - 1e-9))


def test_global_check_fails_on_the_local_example():
    A, B, C, net = nst.example_system("local")
    g = nst.check_global(A, B, C, net)
    assert not g.certified and not g.metzler and "Metzler" in g.reason
    assert g.overshoot is None and g.decay == 0


def test_local_check_on_the_local_example():
    A, B, C, net = nst.example_system("local")
    c = nst.check_local(A, B, C, net, 1.0)
    assert c.certified and c.method == "perron"
    assert c.gamma1.item() == pytest.approx(-2.86175, abs=1e-5)
    assert c.gamma2.item() == pytest.approx(-1.14397, abs=1e-5)
    assert c.decay == pytest.approx(-nst.spectral_abscissa(c.M2), rel=1e-9)
    # C x <= ybar on the region, and the bound is reached at one of its corners
    corners = c.rho / c.v * np.eye(3)
    assert np.all(corners @ C.T <= 1.0 + 1e-12) and np.max(corners @ C.T) == pytest.approx(1.0)
    assert c.contains(0.5 * corners) and not c.contains(1.01 * corners[0]) and not c.contains([-0.1, 0, 0])


def test_lp_region_contains_the_perron_region():
    A, B, C, net = nst.example_system("local")
    per = nst.check_local(A, B, C, net, 1.0)
    lp = nst.check_local(A, B, C, net, 1.0, v="lp", min_decay=per.decay * 0.999)
    assert lp.certified and lp.decay >= per.decay * 0.999 * (1 - 1e-9)
    assert np.all(lp.rho / lp.v >= per.rho / per.v * (1 - 1e-9))     # every corner reaches as far
    lp_half = nst.check_local(A, B, C, net, 1.0, v="lp")               # default: half the best rate
    assert lp_half.decay >= 0.5 * per.decay * (1 - 1e-9)
    assert np.all(lp_half.rho / lp_half.v >= lp.rho / lp.v * (1 - 1e-9))


def test_local_check_fails_on_a_large_box():
    A, B, C, net = nst.example_system("local")
    c = nst.check_local(A, B, C, net, 3.0)
    assert not c.certified and c.metzler and not c.hurwitz and "Hurwitz" in c.reason
    assert c.radius == 0.0 and not c.contains(np.zeros(3))


def test_min_decay():
    A, B, C, net = nst.example_system("local")
    best = nst.check_local(A, B, C, net, 1.0).decay
    assert nst.check_local(A, B, C, net, 1.0, min_decay=0.9 * best).certified
    c = nst.check_local(A, B, C, net, 1.0, min_decay=1.1 * best)
    assert not c.certified and "min_decay" in c.reason


def test_linear_controller_gives_the_linear_test():
    # with no hidden layer, pi(y) = K y and the local sector is exactly [K, K]
    rng = np.random.default_rng(0)
    for _ in range(40):
        A = np.abs(rng.standard_normal((3, 3))) - 2.5 * np.eye(3)
        B, C = np.abs(rng.standard_normal((3, 1))), np.abs(rng.standard_normal((1, 3)))
        K = rng.uniform(-1.5, 0.5)
        c = nst.check_local(A, B, C, ns.Network([[[K]]]), 1.0)
        M = A + K * B @ C
        assert c.certified == (nst.is_metzler(M) and nst.spectral_abscissa(M) < 0)


def test_input_errors():
    A, B, C, net = nst.example_system("local")
    with pytest.raises(ValueError, match="B >= 0"):
        nst.check_global(A, -B, C, net)
    with pytest.raises(ValueError, match="C >= 0"):
        nst.check_global(A, B, -C, net)
    with pytest.raises(ValueError, match="not an equilibrium"):
        nst.check_global(A, B, C, ns.Network(net.weights, [None, None, [0.1]]))
    with pytest.raises(ValueError, match="not an equilibrium"):
        nst.check_global(A, B, C, ns.Network(net.weights, activation="sigmoid"))   # sigmoid(0) = 1/2
    with pytest.raises(ValueError, match="columns"):
        nst.check_global(A, np.ones((3, 2)), C, net)
    with pytest.raises(ValueError, match="rows"):
        nst.check_global(A, B, np.ones((2, 3)), net)
    with pytest.raises(ValueError, match="square"):
        nst.check_global(np.ones((3, 2)), B, C, net)
    with pytest.raises(ValueError, match="ybar"):
        nst.check_local(A, B, C, net, 0.0)
    with pytest.raises(ValueError, match="'perron' or 'lp'"):
        nst.check_local(A, B, C, net, 1.0, v="eig")
    with pytest.raises(TypeError):
        nst.check_global(A, B, C, "not a network")


def test_vectors_for_one_input_and_one_output():
    A, B, C, net = nst.example_system("local")
    a = nst.check_local(A, B, C, net, 1.0)
    b = nst.check_local(A, B.ravel(), C.ravel(), net, [1.0])
    assert np.array_equal(a.v, b.v) and a.rho == b.rho


def test_torch_model_is_accepted():
    torch = pytest.importorskip("torch")
    A, B, C, net = nst.example_system("local")
    model = torch.nn.Sequential(torch.nn.Linear(1, 8, bias=False), torch.nn.Tanh(),
                                torch.nn.Linear(8, 8, bias=False), torch.nn.Tanh(),
                                torch.nn.Linear(8, 1, bias=False)).double()
    with torch.no_grad():
        for layer, W in zip(model[::2], net.weights):
            layer.weight.copy_(torch.as_tensor(W))
    a, b = nst.check_local(A, B, C, net, 1.0), nst.check_local(A, B, C, model, 1.0)
    assert np.allclose(a.v, b.v) and a.rho == pytest.approx(b.rho)
