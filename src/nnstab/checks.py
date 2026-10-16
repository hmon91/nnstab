"""Global and local exponential stability checks of a positive NN feedback loop.

The loop is  x' = A x + B pi(C x)  with B >= 0, C >= 0 and pi(0) = 0, for
initial states x0 >= 0. If pi lies in the sector [S1, S2] on a set of outputs
y >= 0, that is S1 y <= pi(y) <= S2 y, the test is the positive Aizerman
condition:

    M1 = A + B S1 C is Metzler   and   M2 = A + B S2 C is Hurwitz.

Globally, S1 = -Gamma2 and S2 = Gamma2 come from nnsector.global_bound and
the origin is then globally exponentially stable on the nonnegative orthant.
Locally, S1 = gamma1 and S2 = gamma2 come from nnsector.local_bound on the
output box 0 <= y <= ybar. For any v > 0 with v^T M2 < 0, the set

    Omega = {x >= 0 : v^T x <= rho}

with C Omega inside the box is then forward invariant, and every solution
that starts in it decays exponentially: V(x) = v^T x is a copositive
Lyapunov function, and v^T x(t) <= exp(-decay t) v^T x(0).
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Optional

import numpy as np
import nnsector as ns

from .linalg import (copositive_vector, is_hurwitz, is_metzler, largest_region_vector,
                     spectral_abscissa)


def as_network(net) -> ns.Network:
    """``net`` as an nnsector.Network (a torch.nn.Sequential is converted)."""
    if isinstance(net, ns.Network):
        return net
    try:
        import torch
    except ImportError:                       # pragma: no cover
        torch = None
    if torch is not None and isinstance(net, torch.nn.Module):
        return ns.Network.from_torch(net)
    raise TypeError("net must be an nnsector.Network or a torch.nn.Sequential.")


def check_system(A, B, C, net):
    """Check the loop x' = A x + B pi(C x); return A, B, C as float arrays and the network.

    B may be given as a vector for one input, C as a vector for one output.
    """
    net = as_network(net)
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    if B.ndim == 1:
        B = B[:, None]
    if C.ndim == 1:
        C = C[None, :]
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"A must be square, but it is {A.shape[0]} x {A.shape[1]}.")
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError(f"B must have {n} rows, like A.")
    if C.ndim != 2 or C.shape[1] != n:
        raise ValueError(f"C must have {n} columns, like A.")
    if B.shape[1] != net.m:
        raise ValueError(f"B has {B.shape[1]} columns, but the network has {net.m} outputs.")
    if C.shape[0] != net.p:
        raise ValueError(f"C has {C.shape[0]} rows, but the network has {net.p} inputs.")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B)) and np.all(np.isfinite(C))):
        raise ValueError("A, B and C must be finite.")
    if np.any(B < 0) or np.any(C < 0):
        raise ValueError("The method needs B >= 0 and C >= 0 (a positive input and output).")
    pi0 = net(np.zeros(net.p))
    if np.max(np.abs(pi0)) > 1e-10:
        raise ValueError(
            f"The origin is not an equilibrium: pi(0) = {np.array2string(pi0, precision=4)}. "
            "The method needs pi(0) = 0, for example a network without biases with "
            "tanh, ReLU or leaky ReLU activations.")
    return A, B, C, net


@dataclass
class GlobalCheck:
    """Result of :func:`check_global`.

    If ``certified``, every solution with x0 >= 0 satisfies
    ``sum(x(t)) <= overshoot * exp(-decay t) * sum(x0)``.
    """

    Gamma1: np.ndarray              #: lower global sector slope, -Gamma2
    Gamma2: np.ndarray              #: upper global sector slope
    M1: np.ndarray                  #: A + B Gamma1 C
    M2: np.ndarray                  #: A + B Gamma2 C
    metzler: bool                   #: M1 is Metzler
    hurwitz: bool                   #: M2 is Hurwitz
    decay: float                    #: guaranteed decay rate (0 if not certified)
    v: Optional[np.ndarray]         #: copositive vector, v^T M2 <= -decay v^T, min entry 1
    reason: str                     #: why it is not certified ('' if it is)

    @property
    def certified(self) -> bool:
        return self.v is not None

    @property
    def overshoot(self) -> Optional[float]:
        return None if self.v is None else float(self.v.max() / self.v.min())

    def __str__(self) -> str:
        head = f"Global check: {'certified' if self.certified else 'not certified'}"
        lines = [head,
                 f"  global sector slopes  Gamma1 = {_fmt(self.Gamma1)},  Gamma2 = {_fmt(self.Gamma2)}",
                 f"  A + B Gamma1 C Metzler: {_yes(self.metzler)},  A + B Gamma2 C Hurwitz: {_yes(self.hurwitz)}"]
        if self.certified:
            lines.append(f"  every x0 >= 0: sum(x(t)) <= {self.overshoot:.3g} exp(-{self.decay:.3g} t) sum(x0)")
        else:
            lines.append(_wrap(self.reason))
        return "\n".join(lines)


@dataclass
class LocalCheck:
    """Result of :func:`check_local` on the output box ``0 <= y <= ybar``.

    If ``certified``, the region ``{x >= 0 : v^T x <= rho}`` is forward
    invariant, and every solution that starts in it satisfies
    ``v^T x(t) <= exp(-decay t) v^T x0`` and
    ``sum(x(t)) <= overshoot * exp(-decay t) * sum(x0)``.
    """

    ybar: np.ndarray                #: upper corner of the output box
    gamma1: np.ndarray              #: lower local sector slope
    gamma2: np.ndarray              #: upper local sector slope
    M1: np.ndarray                  #: A + B gamma1 C
    M2: np.ndarray                  #: A + B gamma2 C
    metzler: bool                   #: M1 is Metzler
    hurwitz: bool                   #: M2 is Hurwitz
    decay: float                    #: guaranteed decay rate on the region (0 if not certified)
    v: Optional[np.ndarray]         #: copositive vector of the region, min entry 1
    rho: Optional[float]            #: level of the region
    method: str                     #: how v was chosen: 'perron' or 'lp'
    reason: str                     #: why it is not certified ('' if it is)

    @property
    def certified(self) -> bool:
        return self.v is not None

    @property
    def radius(self) -> float:
        """Every x0 >= 0 with sum(x0) <= radius lies in the region (0 if not certified)."""
        return 0.0 if self.v is None else float(self.rho / self.v.max())

    @property
    def overshoot(self) -> Optional[float]:
        return None if self.v is None else float(self.v.max() / self.v.min())

    def contains(self, x, tol: float = 1e-12) -> bool:
        """True if the state x (or every row of x) lies in the certified region."""
        if self.v is None:
            return False
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return bool(np.all(x >= -tol) and np.all(x @ self.v <= self.rho * (1 + tol) + tol))

    def __str__(self) -> str:
        head = f"Local check on 0 <= y <= {_fmt(self.ybar)}: {'certified' if self.certified else 'not certified'}"
        lines = [head,
                 f"  local sector slopes  gamma1 = {_fmt(self.gamma1)},  gamma2 = {_fmt(self.gamma2)}",
                 f"  A + B gamma1 C Metzler: {_yes(self.metzler)},  A + B gamma2 C Hurwitz: {_yes(self.hurwitz)}"]
        if self.certified:
            lines += [f"  region: {{x >= 0 : v^T x <= {self.rho:.4g}}},  v = {_fmt(self.v)}",
                      f"  it holds every x0 >= 0 with sum(x0) <= {self.radius:.4g}",
                      f"  in it: sum(x(t)) <= {self.overshoot:.3g} exp(-{self.decay:.3g} t) sum(x0)"]
        else:
            lines.append(_wrap(self.reason))
        return "\n".join(lines)


def check_global(A, B, C, net) -> GlobalCheck:
    """Global test: the positive Aizerman condition with the global sector bound.

    Gamma2 = c^q |W_{q+1}| ... |W_1| (nnsector.global_bound) and
    Gamma1 = -Gamma2. Certified if A + B Gamma1 C is Metzler and
    A + B Gamma2 C is Hurwitz; then the origin is globally exponentially
    stable for all x0 >= 0.
    """
    A, B, C, net = check_system(A, B, C, net)
    G2 = ns.global_bound(net).Gamma2
    G1 = -G2
    M1, M2 = A + B @ G1 @ C, A + B @ G2 @ C
    metzler, hurwitz = is_metzler(M1), is_hurwitz(M2)
    v, decay, reason = None, 0.0, ""
    if metzler and hurwitz:
        v, decay = copositive_vector(M2)
    elif not metzler:
        reason = ("A + B Gamma1 C is not Metzler: the global lower slope -Gamma2 is too "
                  "negative to keep the states nonnegative.")
    else:
        reason = "A + B Gamma2 C is not Hurwitz: the global upper slope Gamma2 is too large."
    return GlobalCheck(G1, G2, M1, M2, metzler, hurwitz, decay, v, reason)


def check_local(A, B, C, net, ybar, v: str = "perron", min_decay: float = 0.0) -> LocalCheck:
    """Local test on the output box 0 <= y <= ybar, with its region of attraction.

    gamma1, gamma2 come from nnsector.local_bound on the box, about y* = 0.
    Certified if A + B gamma1 C is Metzler and A + B gamma2 C is Hurwitz;
    then the region {x >= 0 : v^T x <= rho} is invariant and attracting,
    where rho is the largest level with C x <= ybar on the region.

    Parameters
    ----------
    ybar : float or array of shape (p,)
        Upper corner of the output box (all entries > 0).
    v : {'perron', 'lp'}
        How the copositive vector is chosen. 'perron': the left eigenvector
        of A + B gamma2 C for its largest eigenvalue, which gives the fastest
        guaranteed decay. 'lp': the vector that makes the region as large as
        possible while keeping the decay rate at least ``min_decay`` (or,
        if min_decay is 0, half of the best rate).
    min_decay : float
        Smallest decay rate to accept (default 0: any rate).
    """
    A, B, C, net = check_system(A, B, C, net)
    p = net.p
    ybar = np.asarray(ybar, dtype=float)
    ybar = np.full(p, float(ybar.reshape(-1)[0])) if ybar.size == 1 else ybar.reshape(-1)
    if ybar.shape != (p,) or not np.all(np.isfinite(ybar)) or np.any(ybar <= 0):
        raise ValueError(f"ybar must be a positive scalar or have {p} positive entries.")
    if v not in ("perron", "lp"):
        raise ValueError("v must be 'perron' or 'lp'.")
    if not min_decay >= 0:
        raise ValueError("min_decay must be >= 0.")

    sb = ns.local_bound(net, np.zeros(p), ybar, ystar=np.zeros(p))
    g1, g2 = sb.gamma1, sb.gamma2
    M1, M2 = A + B @ g1 @ C, A + B @ g2 @ C
    metzler, hurwitz = is_metzler(M1), is_hurwitz(M2)

    def fail(reason):
        return LocalCheck(ybar, g1, g2, M1, M2, metzler, hurwitz, 0.0, None, None, v, reason)

    if not metzler:
        return fail("A + B gamma1 C is not Metzler: the lower slope gamma1 is too negative "
                    "to keep the states nonnegative.")
    if not hurwitz:
        return fail("A + B gamma2 C is not Hurwitz: the upper slope gamma2 is too large.")
    best = -spectral_abscissa(M2)
    if min_decay > 0 and min_decay >= best:
        return fail(f"The best possible decay rate here is {best:.3g}, not above min_decay = {min_decay:g}.")

    if v == "perron":
        vec, decay = copositive_vector(M2)
        kappa = (C / vec[None, :]).max(axis=1)           # largest C_ji / v_i, per output j
        hit = kappa > 0
        rho = float(np.min(ybar[hit] / kappa[hit])) if hit.any() else np.inf
    else:
        rate = min_decay if min_decay > 0 else 0.5 * best
        c = (C / ybar[:, None]).max(axis=0)              # C x <= ybar on the region iff w >= c
        c = np.maximum(c, 1e-12 * max(c.max(), 1.0))     # keep w > 0 so the region is bounded
        w = largest_region_vector(M2, c, rate)
        decay = float(np.min(-(w @ M2) / w))
        vec, rho = w / w.min(), float(1.0 / w.min())
    if not decay > 0 or (min_decay > 0 and decay < min_decay * (1 - 1e-9)):
        return fail("No copositive vector with the requested decay rate was found.")
    return LocalCheck(ybar, g1, g2, M1, M2, metzler, hurwitz, float(decay), vec, rho, v, "")


# ---- helpers -----------------------------------------------------------------

def _wrap(text: str, first: str = "  ") -> str:
    """Text for a report: indented, with lines of at most 88 characters."""
    return textwrap.fill(text, width=88, initial_indent=first, subsequent_indent="  ")


def _fmt(a) -> str:
    """A number, a vector or a matrix in short form, such as [1.929, 2.089, 1]."""
    a = np.asarray(a, dtype=float)
    if a.size == 1:
        return f"{a.item():.4g}"
    if a.ndim == 1 or 1 in a.shape:
        return "[" + ", ".join(f"{x:.4g}" for x in a.ravel()) + "]"
    return "[" + ", ".join(_fmt(row) for row in a) + "]"


def _yes(b: bool) -> str:
    return "yes" if b else "no"
