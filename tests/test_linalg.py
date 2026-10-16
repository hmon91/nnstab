"""The matrix tests: Metzler, Hurwitz certificates, Perron vectors, the region vector."""

import numpy as np
import pytest

import nnstab as nst


def random_metzler_hurwitz(rng, n):
    M = np.abs(rng.standard_normal((n, n)))
    np.fill_diagonal(M, 0.0)
    shift = np.max(np.linalg.eigvals(M).real) + rng.uniform(0.1, 2.0)
    return M - shift * np.eye(n)                    # spectral abscissa = -(shift - rho(M)) < 0


def test_is_metzler():
    assert nst.is_metzler([[-1.0, 0.0], [2.0, -3.0]])
    assert not nst.is_metzler([[-1.0, -0.1], [2.0, -3.0]])
    assert nst.is_metzler([[5.0, 1e-20], [-1e-14, 3.0]])     # rounding-size negatives are allowed


def test_hurwitz_certificate():
    rng = np.random.default_rng(0)
    for _ in range(50):
        M = random_metzler_hurwitz(rng, int(rng.integers(1, 7)))
        v = nst.hurwitz_certificate(M)
        assert v is not None and np.all(v > 0) and np.allclose(v @ M, -1.0)
        assert nst.hurwitz_certificate(M + (0.01 - nst.spectral_abscissa(M)) * np.eye(len(M))) is None


def test_is_hurwitz():
    rng = np.random.default_rng(2)
    for _ in range(30):
        M = random_metzler_hurwitz(rng, int(rng.integers(1, 6)))
        assert nst.is_hurwitz(M) and not nst.is_hurwitz(M + (0.01 - nst.spectral_abscissa(M)) * np.eye(len(M)))
    assert nst.is_hurwitz([[-1.0, -5.0], [5.0, -1.0]])          # not Metzler: eigenvalues -1 +- 5i
    assert not nst.is_hurwitz([[1.0, -5.0], [5.0, -1.0]])


def test_copositive_vector():
    rng = np.random.default_rng(3)
    for _ in range(30):
        M = random_metzler_hurwitz(rng, int(rng.integers(1, 6)))
        v, d = nst.copositive_vector(M)
        assert v.min() == pytest.approx(1.0) and d == pytest.approx(-nst.spectral_abscissa(M))
        assert np.all(v @ M <= -d * v * (1 - 1e-9))
    # reducible: the Perron vector has a zero entry, so the certificate is used
    M = np.array([[-1.0, 0.0], [1.0, -2.0]])
    v, d = nst.copositive_vector(M)
    assert np.all(v > 0) and np.all(v @ M <= -d * v) and 0 < d <= 1.0
    with pytest.raises(ValueError, match="Metzler and Hurwitz"):
        nst.copositive_vector([[1.0, 0.0], [0.0, -1.0]])


def test_perron_vector():
    rng = np.random.default_rng(1)
    for _ in range(50):
        M = random_metzler_hurwitz(rng, int(rng.integers(1, 7)))
        v = nst.perron_vector(M)
        assert v is not None and v.min() == pytest.approx(1.0)
        assert np.allclose(v @ M, nst.spectral_abscissa(M) * v)


@pytest.mark.parametrize("seed", range(40))
def test_largest_region_vector_is_the_least_feasible_vector(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 7))
    M = random_metzler_hurwitz(rng, n)
    c = np.abs(rng.standard_normal(n)) * (rng.random(n) < 0.8)
    rate = rng.uniform(0.0, 0.9) * -nst.spectral_abscissa(M)
    w = nst.largest_region_vector(M, c, rate)
    s = w @ (M + rate * np.eye(n))
    scale = 1 + np.abs(w).max()
    assert np.all(w >= c - 1e-12 * scale) and np.all(s <= 1e-10 * scale)       # feasible
    assert np.allclose((w - c) * s, 0.0, atol=1e-9 * scale ** 2)                # complementary
    # complementarity of the M-matrix problem makes w the least feasible vector:
    # every other feasible vector, e.g. a scaled Perron vector, is >= w
    v = nst.perron_vector(M)
    t = max(1.0, np.max(c / v))
    assert np.all(t * v >= w - 1e-9 * scale)


def test_largest_region_vector_against_linprog():
    linprog = pytest.importorskip("scipy.optimize").linprog
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = int(rng.integers(2, 6))
        M = random_metzler_hurwitz(rng, n)
        c = np.abs(rng.standard_normal(n))
        rate = 0.3 * -nst.spectral_abscissa(M)
        w = nst.largest_region_vector(M, c, rate)
        lp = linprog(np.ones(n), A_ub=(M + rate * np.eye(n)).T, b_ub=np.zeros(n),
                     bounds=[(ci, None) for ci in c])
        assert lp.success and np.allclose(w, lp.x, rtol=1e-6, atol=1e-8)
