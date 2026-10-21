"""Matrix tests behind the stability checks.

For a Metzler matrix M (all off-diagonal entries >= 0) the following are
equivalent: M is Hurwitz; there is a vector v > 0 with v^T M < 0. Such a v
is a copositive certificate: V(x) = v^T x decreases along x' = M x on the
nonnegative orthant.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def is_metzler(M, tol: float = 1e-12) -> bool:
    """True if every off-diagonal entry of M is >= 0 (up to ``tol`` times the scale of M)."""
    M = np.asarray(M, dtype=float)
    off = M - np.diag(np.diag(M))
    return bool(np.all(off >= -tol * max(1.0, np.abs(M).max())))


def spectral_abscissa(M) -> float:
    """Largest real part of the eigenvalues of M."""
    return float(np.max(np.linalg.eigvals(np.asarray(M, dtype=float)).real))


def hurwitz_certificate(M) -> Optional[np.ndarray]:
    """For a Metzler M: a vector v > 0 with v^T M = -1^T, or None if there is none.

    Such a v exists exactly when M is Hurwitz, since then -M is a nonsingular
    M-matrix and v = (-M^T)^{-1} 1 > 0. The result is checked, so a returned v
    is a certificate on its own.
    """
    M = np.asarray(M, dtype=float)
    try:
        v = np.linalg.solve(-M.T, np.ones(M.shape[0]))
    except np.linalg.LinAlgError:
        return None
    if np.all(np.isfinite(v)) and np.all(v > 0) and np.all(v @ M < 0):
        return v
    return None


def is_hurwitz(M) -> bool:
    """True if every eigenvalue of M has a negative real part.

    For a Metzler M the answer comes from :func:`hurwitz_certificate`, which
    is a direct test and does not rely on computed eigenvalues.
    """
    if is_metzler(M):
        return hurwitz_certificate(M) is not None
    return spectral_abscissa(M) < 0


def perron_vector(M) -> Optional[np.ndarray]:
    """Left eigenvector of a Metzler M for its eigenvalue with the largest real part.

    Scaled so that its smallest entry is 1. Returns None if it is not
    strictly positive (this can happen when M is reducible).
    """
    M = np.asarray(M, dtype=float)
    lam, V = np.linalg.eig(M.T)
    k = int(np.argmax(lam.real))
    v = np.real(V[:, k])
    v = v * np.sign(v[np.argmax(np.abs(v))])
    if not np.all(np.isfinite(v)) or np.min(v) <= 1e-12 * np.max(v):
        return None
    return v / v.min()


def copositive_vector(M):
    """For a Metzler, Hurwitz M: a vector v > 0 and a rate d > 0 with v^T M <= -d v^T.

    v is the left Perron vector of M (smallest entry 1), which gives the
    best rate, d = -spectral_abscissa(M). If that vector is not strictly
    positive (this can happen when M is reducible), the certificate
    (-M^T)^{-1} 1 is used instead. Returns (v, d).
    """
    M = np.asarray(M, dtype=float)
    v = perron_vector(M)
    if v is not None:
        d = float(np.min(-(v @ M) / v))
        if d > 0:
            return v, d
    v = hurwitz_certificate(M)
    if v is None:
        raise ValueError("M must be Metzler and Hurwitz.")
    v = v / v.min()
    return v, float(np.min(-(v @ M) / v))


def largest_region_vector(M, c, rate: float) -> np.ndarray:
    """The least vector w with  w >= c  and  w^T (M + rate I) <= 0.

    M must be Metzler with M + rate I Hurwitz, and c >= 0. Then
    Z = -(M + rate I)^T is a nonsingular M-matrix, the feasible set is closed
    under the componentwise minimum, and its least element is the solution
    of the linear complementarity problem  z = w - c >= 0,  s = Z w >= 0,
    z_i s_i = 0. It is found by Chandrasekaran's method in at most n solves.
    The least w gives the largest region {x >= 0 : w^T x <= 1} among all
    such vectors: it minimizes every entry of w at once.
    """
    M = np.asarray(M, dtype=float)
    c = np.asarray(c, dtype=float)
    n = len(c)
    Z = -(M + rate * np.eye(n)).T
    scale = max(1.0, np.abs(Z).max())
    basis = np.zeros(n, dtype=bool)
    for _ in range(n + 1):
        z = np.zeros(n)
        if basis.any():
            z[basis] = np.linalg.solve(Z[np.ix_(basis, basis)], -(Z @ c)[basis])
        w = c + z
        s = Z @ w
        new = (~basis) & (s < -1e-12 * scale * max(1.0, np.abs(w).max()))
        if not new.any():
            return w
        basis |= new
    raise RuntimeError("largest_region_vector did not converge; is M + rate I Hurwitz?")
