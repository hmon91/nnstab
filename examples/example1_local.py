"""The local example: the global test fails, and a region of attraction is found.

The plant is unstable on its own. The controller is a small tanh network
without biases (nnstab.example_controller()). Near y = 0 it pushes back
hard, which makes the loop stable. For large y its output levels off, the
feedback gets too weak, and the loop is unstable. So only a local result is
possible.

The certificate is then checked by simulation, from random starting states
in the region. Plots them if matplotlib is installed.
"""

import numpy as np

import nnstab as nst

A, B, C, net = nst.example_system("local")
print("eigenvalues of A:", np.round(np.linalg.eigvals(A).real, 3))
print()

res = nst.verify(A, B, C, net)                   # the largest region
print(res)
print()

fast = nst.verify(A, B, C, net, min_decay=0.5)   # a smaller region, decay rate at least 0.5
print(fast)
print()

# simulate from 200 random starting states in the region of `fast`
c = fast.local_check
X0 = nst.sample_region(c, 200, seed=1)
t, X = nst.simulate(A, B, C, net, X0, t_end=8.0)
V = X @ c.v                                      # V(x(t)) = v^T x(t), one column per solution
bound = V[0] * np.exp(-c.decay * t)[:, None]     # the guarantee
print("Simulation of 200 solutions that start in the region:")
print("  every state stays >= 0           :", bool(X.min() >= -1e-12))
print("  every solution stays in it       :", bool(np.all(V <= c.rho * (1 + 1e-9))))
print("  v^T x(t) <= exp(-decay t) v^T x0 :", bool(np.all(V <= bound * (1 + 1e-9) + 1e-12)))

# outside the region nothing is promised, and far away the loop is unstable
_, Xfar = nst.simulate(A, B, C, net, [4.0, 4.0, 4.0], t_end=8.0)
print(f"  from x0 = (4, 4, 4), outside it  : sum(x(8)) = {Xfar[-1].sum():.3g}")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("(install matplotlib to see the plot)")
else:
    fig, ax = plt.subplots()
    ax.semilogy(t, V / V[0], color="#2a78d6", lw=0.6, alpha=0.35)
    ax.plot([], [], color="#2a78d6", lw=1.5, label="200 solutions")
    ax.semilogy(t, np.exp(-c.decay * t), "k--", lw=2,
                label=rf"guarantee $e^{{-{c.decay:.2f}\,t}}$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$v^T x(t) \;/\; v^T x_0$")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    plt.show()
