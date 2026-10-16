"""One call that checks global stability first and then looks for a local region."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .checks import GlobalCheck, LocalCheck, _fmt, _wrap, check_global, check_local, check_system


@dataclass
class Verification:
    """Result of :func:`verify`.

    ``status`` is 'global' (globally exponentially stable for every x0 >= 0),
    'local' (exponentially stable on the region of ``local_check``) or
    'none' (neither test could be passed; this does not prove instability).
    """

    status: str
    global_check: GlobalCheck
    local_check: Optional[LocalCheck]
    boxes_tried: int = 0
    note: str = ""

    @property
    def stable(self) -> bool:
        """True if stability was certified, globally or locally."""
        return self.status != "none"

    def contains(self, x) -> bool:
        """True if the certificate covers the initial state x (or every row of x)."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if self.status == "global":
            return bool(np.all(x >= 0))
        if self.status == "local":
            return self.local_check.contains(x)
        return False

    def __str__(self) -> str:
        if self.status == "global":
            g = self.global_check
            lines = ["Globally exponentially stable: every solution with x0 >= 0 goes to 0.",
                     f"  sum(x(t)) <= {g.overshoot:.3g} exp(-{g.decay:.3g} t) sum(x0)"]
        elif self.status == "local":
            c = self.local_check
            lines = ["Locally exponentially stable (the global test failed).",
                     f"  region: {{x >= 0 : v^T x <= {c.rho:.4g}}},  v = {_fmt(c.v)}",
                     f"  it holds every x0 >= 0 with sum(x0) <= {c.radius:.4g}",
                     f"  in it: sum(x(t)) <= {c.overshoot:.3g} exp(-{c.decay:.3g} t) sum(x0)",
                     f"  output box 0 <= y <= {_fmt(c.ybar)}"]
        else:
            lines = ["Not verified: neither the global nor a local test passed.",
                     "  (This does not mean the loop is unstable.)"]
        if self.note:
            lines.append(_wrap(self.note, first="  note: "))
        lines += ["", str(self.global_check)]
        if self.local_check is not None:
            lines += ["", str(self.local_check)]
        return "\n".join(lines)


def verify(A, B, C, net, ybar=None, direction=None, v: str = "perron",
           min_decay: float = 0.0, rtol: float = 1e-3) -> Verification:
    """Check exponential stability of x' = A x + B pi(C x) for x0 >= 0.

    First the global test (:func:`check_global`). If it fails, the local test
    (:func:`check_local`) on the output box 0 <= y <= ybar. Without ``ybar``,
    the box ``s * direction`` is searched over s > 0 for the largest
    certified region: the box is grown (or shrunk) by factors of 2 until the
    test changes its answer, the edge is found by bisection to relative
    accuracy ``rtol``, and the largest region among a few boxes below the
    edge is kept.

    Parameters
    ----------
    A, B, C : arrays
        The plant x' = A x + B u, y = C x, with B >= 0 and C >= 0.
    net : nnsector.Network or torch.nn.Sequential
        The controller u = pi(y), with pi(0) = 0.
    ybar : float or array of shape (p,), optional
        Check only this output box.
    direction : array of shape (p,), optional
        Shape of the searched boxes (default: all ones).
    v, min_decay
        Passed to :func:`check_local`. Near the edge of the largest region
        the guaranteed decay rate gets small; ask for a faster one with
        ``min_decay``.
    """
    A, B, C, net = check_system(A, B, C, net)
    g = check_global(A, B, C, net)
    if g.certified:
        return Verification("global", g, None)
    if ybar is not None:
        loc = check_local(A, B, C, net, ybar, v=v, min_decay=min_decay)
        return Verification("local" if loc.certified else "none", g, loc, 1)
    p = net.p
    d = np.ones(p) if direction is None else np.asarray(direction, dtype=float).reshape(-1)
    if d.shape != (p,) or not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError(f"direction must have {p} positive entries.")
    best, tried, note = _search(lambda s: check_local(A, B, C, net, s * d, v=v, min_decay=min_decay), rtol)
    if best.certified and not note and min_decay == 0:
        note = ("this is the largest region found, and near its edge the guaranteed decay "
                "rate is small; pass min_decay for a faster guaranteed rate on a smaller region.")
    return Verification("local" if best.certified else "none", g, best, tried, note)


def _search(check, rtol, max_scale=2.0 ** 40, min_scale=2.0 ** -40):
    """Find the box scale s with the largest certified region; returns (check, tries, note)."""
    tried = 0

    def run(s):
        nonlocal tried
        tried += 1
        return check(s)

    s = 1.0
    first = run(s)
    if first.certified:
        lo, lo_chk, hi = s, first, None
        while s < max_scale:
            s *= 2.0
            c = run(s)
            if not c.certified:
                hi = s
                break
            lo, lo_chk = s, c
        if hi is None:
            return lo_chk, tried, ("the local test still passes on very large boxes; "
                                   "the loop may be globally stable, but the global test is too conservative.")
    else:
        hi, lo_chk, last = s, None, first
        while s > min_scale:
            s /= 2.0
            c = run(s)
            if c.certified:
                lo, lo_chk = s, c
                break
            hi, last = s, c
        if lo_chk is None:
            return last, tried, "the local test fails even on very small boxes."
    while hi - lo > rtol * hi:
        mid = 0.5 * (lo + hi)
        c = run(mid)
        if c.certified:
            lo, lo_chk = mid, c
        else:
            hi = mid
    best = lo_chk
    for s in lo * np.linspace(0.5, 1.0, 6)[:-1]:     # the region need not be largest at the edge
        c = run(s)
        if c.certified and c.radius > best.radius:
            best = c
    return best, tried, ""
