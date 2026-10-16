"""The global example: the global test passes.

A stable plant that the controller only nudges: the controller sees a small
part of the state (small C), and its input enters through small entries of
B. Then even the wide global sector of the network is narrow enough, and
the loop is exponentially stable for every starting state x0 >= 0.
"""

import numpy as np

import nnstab as nst

A, B, C, net = nst.example_system("global")
res = nst.verify(A, B, C, net)
print(res)
print()

# simulate from 100 large random starting states
g = res.global_check
X0 = np.random.default_rng(0).exponential(scale=100.0, size=(100, 3))
t, X = nst.simulate(A, B, C, net, X0, t_end=10.0)
S = X.sum(axis=2)                                # sum(x(t)), one column per solution
bound = g.overshoot * np.exp(-g.decay * t)[:, None] * S[0]
print("Simulation of 100 solutions from large starting states:")
print("  every state stays >= 0                        :", bool(X.min() >= -1e-12))
print("  sum(x(t)) <= overshoot exp(-decay t) sum(x0)  :", bool(np.all(S <= bound * (1 + 1e-9))))
print(f"  largest sum(x0) and sum(x(10))                : {S[0].max():.4g}, {S[-1].max():.3g}")
