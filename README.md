# nnstab

[![tests](https://github.com/hmon91/nnstab/actions/workflows/tests.yml/badge.svg)](https://github.com/hmon91/nnstab/actions/workflows/tests.yml)

**nnstab** verifies the exponential stability of positive feedback systems
with a neural network controller. It is implemented in Python and in MATLAB
(compatible with GNU Octave), and it uses sector bounds of the network
computed with [nnsector](https://github.com/hmon91/nnsector).

## Problem statement

nnstab considers the closed-loop system

```math
\dot{x}(t) = A\,x(t) + B\,u(t), \qquad y(t) = C\,x(t), \qquad u(t) = \pi\big(y(t)\big),
```

with state $`x \in \mathbb{R}^n`$, output $`y \in \mathbb{R}^p`$, input
$`u \in \mathbb{R}^m`$, and a trained feedforward neural network
$`\pi : \mathbb{R}^p \to \mathbb{R}^m`$. The system is assumed to be a
positive Lur'e system, that is:

* **(A1)** $`B \ge 0`$ and $`C \ge 0`$ (entrywise);
* **(A2)** $`\pi(0) = 0`$, so that the origin is an equilibrium. This holds,
  for example, for networks without biases and with tanh, ReLU or leaky ReLU
  activation functions.

The matrix $`A`$ need not be Hurwitz. nnstab checks (A1) and (A2) and reports
an error if either is violated. As usual for positive systems, the state
space is the nonnegative orthant, so the results below concern initial
states $`x_0 \ge 0`$.

The verification has two stages:

1. **Global stage.** nnstab tests whether the origin is globally
   exponentially stable, that is, for every initial state $`x_0 \ge 0`$.
2. **Local stage.** If global stability cannot be established, nnstab
   computes an estimate of the region of attraction: a set of initial states
   from which every solution converges exponentially to the origin. It
   searches for the largest such set.

## Installation

```bash
pip install git+https://github.com/hmon91/nnstab.git
```

This command also installs nnsector. The package requires Python 3.10 or
later and NumPy. PyTorch is optional and is needed only for PyTorch models
(`pip install torch`). To install the package for development and run the
tests:

```bash
git clone https://github.com/hmon91/nnstab.git
cd nnstab
pip install -e ".[test]"
pytest
```

## Quick start

```python
import nnstab as nst

A, B, C, net = nst.example_system("local")     # or a user-defined system
res = nst.verify(A, B, C, net)
print(res)
```

The first lines of the output are:

```
Locally exponentially stable (the global test failed).
  region: {x >= 0 : v^T x <= 3.563},  v = [1.929, 2.089, 1]
  it holds every x0 >= 0 with sum(x0) <= 1.706
  in it: sum(x(t)) <= 2.09 exp(-0.000324 t) sum(x0)
  output box 0 <= y <= 1.848
```

The full report continues with the details of the global and the local
test. The lines shown above have the following meaning.

| Line | Meaning |
|---|---|
| `region` | The certified region of attraction $`\Omega = \{x \ge 0 : v^\top x \le \rho\}`$. |
| `sum(x0) <= 1.706` | Every initial state $`x_0 \ge 0`$ with $`\lVert x_0 \rVert_1 \le 1.706`$ lies in $`\Omega`$. This number is a simple measure of the size of $`\Omega`$. |
| `sum(x(t)) <= ...` | The guaranteed exponential bound for solutions that start in $`\Omega`$, with overshoot constant 2.09 and decay rate 0.000324. |
| `output box` | The box of outputs on which the local sector bound was computed. |

The largest region has a small guaranteed decay rate, because its box lies
at the limit of what the test allows. A larger guaranteed decay rate is
obtained on a smaller region with the option `min_decay`:

```python
res = nst.verify(A, B, C, net, min_decay=0.5)
# The region contains every x0 >= 0 with sum(x0) <= 1.32,
# and sum(x(t)) <= 2.1 exp(-0.501 t) sum(x0) holds on it.
```

The result can also be used in code:

```python
res.status          # "global", "local" or "none"
res.stable          # True if the global or the local test succeeded
res.contains(x0)    # True if the result covers the initial state x0
res.global_check    # details of the global test (GlobalCheck)
res.local_check     # details of the local test (LocalCheck; None after a global result)
```

**User-defined systems.**

```python
import numpy as np
import nnsector as ns
import nnstab as nst

A = np.array([[-3.0,  3.5,  2.0],
              [ 2.0, -4.0,  1.0],
              [ 3.5,  3.5, -5.0]])
B = np.array([[1.0], [0.5], [1.0]])
C = np.array([[1.0, 1.0, 0.5]])
net = ns.Network([W1, W2, W3], activation="tanh")   # no biases, hence pi(0) = 0

res = nst.verify(A, B, C, net)
```

A trained PyTorch model can be passed in place of `net`, for example a
`torch.nn.Sequential` of `nn.Linear(..., bias=False)` layers and `nn.Tanh()`
activations. It is converted with `nnsector.Network.from_torch`.

## Method

**Sector condition.** Suppose that the network satisfies, entrywise,

```math
\Gamma_1\,y \;\le\; \pi(y) \;\le\; \Gamma_2\,y \qquad \text{for all } y \in \mathcal{Y},
```

where $`\mathcal{Y}`$ is a set of nonnegative outputs, and consider the two
conditions

* **(C1)** $`A + B\Gamma_1 C`$ is **Metzler**, that is, all its off-diagonal
  entries are nonnegative (this keeps the state nonnegative);
* **(C2)** $`A + B\Gamma_2 C`$ is **Hurwitz**, that is, all its eigenvalues
  have negative real parts.

If $`\mathcal{Y}`$ is the whole nonnegative orthant, (C1) and (C2) imply that
the origin is globally exponentially stable for all $`x_0 \ge 0`$; this is
the positive Aizerman result [[3]](#citation). If $`\mathcal{Y}`$ is a box,
the same conditions give exponential stability for solutions whose outputs
remain in the box. nnstab uses the first case in the global test and the
second case in the local test.

**Global test.** nnsector provides the global bound
$`\Gamma_2 = c^{\,q}\,\lvert W_{q+1} \rvert \cdots \lvert W_1 \rvert`$ and
$`\Gamma_1 = -\Gamma_2`$, which hold for all $`y \ge 0`$. If (C1) and (C2)
hold for these matrices, the origin is globally exponentially stable for all
$`x_0 \ge 0`$ [[2]](#citation). The global sector is wide, so this test is
conservative. It typically succeeds when the network has a small effect on
the loop.

**Local test.** On a box of outputs $`0 \le y \le \bar{y}`$, nnsector
provides slopes $`\gamma_1, \gamma_2`$ that are much closer together. If (C1)
and (C2) hold with these slopes, that is, if $`A + B\gamma_1 C`$ is Metzler
and $`M = A + B\gamma_2 C`$ is Hurwitz, there is a vector $`v > 0`$ with
$`v^\top M < 0`$, and nnstab constructs the set

```math
\Omega = \{\, x \ge 0 \;:\; v^\top x \le \rho \,\},
```

where $`\rho`$ is the largest value for which $`C x \le \bar{y}`$ holds for
all $`x \in \Omega`$. The function $`V(x) = v^\top x`$ is a linear copositive
Lyapunov function. The set $`\Omega`$ is forward invariant, and every
solution that starts in $`\Omega`$ satisfies

```math
v^\top x(t) \;\le\; e^{-\lambda t}\, v^\top x_0 ,
\qquad
\lVert x(t) \rVert_1 \;\le\; \frac{v_{\max}}{v_{\min}}\, e^{-\lambda t}\, \lVert x_0 \rVert_1 ,
```

where $`\lambda > 0`$ is the guaranteed decay rate (`decay`) and
$`v_{\max}/v_{\min}`$, the ratio of the largest to the smallest entry of
$`v`$, is the overshoot constant (`overshoot`). The local test combines local
sector bounds with a localized form of the positive Aizerman conditions, as
in [[1]](#citation).

**Search over boxes.** A larger box allows a larger region, but its slopes
are farther apart, and beyond some size the test usually fails. Unless a box
is specified, nnstab considers the boxes $`0 \le y \le s\,d`$ for a fixed
direction $`d > 0`$ (by default, all ones). It doubles or halves $`s`$ until
the outcome of the test changes, locates the boundary by bisection, and
returns the box whose region contains the largest set
$`\{x_0 \ge 0 : \lVert x_0 \rVert_1 \le r\}`$.

**Choice of $`v`$.** By default (`v="perron"`), $`v`$ is the left
eigenvector of $`M`$ for its eigenvalue with the largest real part (the
Perron eigenvector). This choice gives the largest possible guaranteed decay
rate. If this eigenvector is not strictly positive, which can happen when
$`M`$ is reducible, the vector $`v = (-M^\top)^{-1}\mathbf{1}`$ is used
instead. With `v="lp"`, $`v`$ is chosen to make the region as large as possible
in every coordinate direction, subject to a decay rate of at least
`min_decay` (or half of the largest rate if `min_decay` is 0). This is a
linear program. Because of its structure, nnstab solves it exactly as a
linear complementarity problem with an M-matrix, in at most $`n`$ linear
solves and without an LP solver.

**Interpretation.** Both tests are sufficient conditions. If a test
succeeds, stability is certified. If both tests fail, the closed loop may
still be stable.

## Options of `verify`

`verify(A, B, C, net, ybar=None, direction=None, v="perron", min_decay=0.0, rtol=1e-3)`

| Argument | Description |
|---|---|
| `ybar` | Tests only the box $`0 \le y \le \bar{y}`$ (a scalar, or one value per output) instead of searching. |
| `direction` | The direction $`d`$ of the searched boxes $`0 \le y \le s\,d`$ (default: all ones). |
| `v` | `"perron"` (default): largest guaranteed decay rate. `"lp"`: largest region for a prescribed decay rate. |
| `min_decay` | Smallest acceptable decay rate (default 0). Larger values give smaller regions. |
| `rtol` | Relative accuracy of the bisection (default `1e-3`). |

## Functions

| Function | Description |
|---|---|
| `verify(A, B, C, net, ...)` | Global test, followed by the search for a local region if the global test fails. Returns a `Verification`. |
| `check_global(A, B, C, net)` | Global test only. Returns a `GlobalCheck`. |
| `check_local(A, B, C, net, ybar, v="perron", min_decay=0)` | Local test on one box. Returns a `LocalCheck`. |
| `simulate(A, B, C, net, x0, t_end, dt=0.01)` | Simulates the closed loop (fourth-order Runge-Kutta method) from one or several initial states. |
| `sample_region(check, k, seed=0)` | Draws `k` initial states uniformly from a certified region, for validation by simulation. |
| `example_system(kind)` | The example systems `"local"` and `"global"`. Returns `A, B, C, net`. |
| `example_controller()` | The example network: one input, two hidden layers of eight tanh neurons, one output, and no biases. |
| `is_metzler`, `is_hurwitz`, `spectral_abscissa`, `hurwitz_certificate`, `perron_vector`, `copositive_vector`, `largest_region_vector` | Matrix routines used by the tests. |

The network may be an `nnsector.Network` or a `torch.nn.Sequential`. For one
input, `B` may be given as a vector; for one output, `C` may be given as a
vector.

The result objects contain the following fields:

* `GlobalCheck`: `Gamma1`, `Gamma2`, `M1` (the matrix $`A + B\Gamma_1 C`$),
  `M2` (the matrix $`A + B\Gamma_2 C`$), `metzler`, `hurwitz`, `certified`,
  `decay`, `v`, `overshoot`, and `reason` (the cause of a failure).
* `LocalCheck`: the same fields with `gamma1` and `gamma2`, together with
  `ybar`, `rho`, `radius` (every initial state $`x_0 \ge 0`$ whose entries
  sum to at most `radius` lies in the region), `method`, and the method
  `contains(x)`.
* `Verification`: `status`, `stable`, `global_check`, `local_check`,
  `boxes_tried`, `note`, and the method `contains(x)`.

Printing any of these objects gives a short report.

## Example systems

The example controller (`example_controller()`) was trained to approximate
$`u = -\tanh(2y) - 0.5\,y`$ on $`0 \le y \le 3`$. It has no biases, so
$`\pi(0) = 0`$. Near the origin, $`\pi(y) \approx -2.5\,y`$. For large $`y`$
its output saturates, and the gain $`\lvert \pi(y)/y \rvert`$ tends to zero.

* `example_system("local")`: the matrix $`A`$ is unstable (it has the
  eigenvalue 0.95). The controller stabilizes the loop near the origin but
  not for large states; for example, the solution from
  $`x_0 = (4, 4, 4)`$ diverges. Global stability therefore does not hold, and
  nnstab returns a region of attraction that contains every $`x_0 \ge 0`$
  with $`\lVert x_0 \rVert_1 \le 1.706`$.
* `example_system("global")`: a stable plant with small input and output
  matrices. The global test succeeds.

## Examples and tests

| File | Content |
|---|---|
| `examples/example1_local.py` | The local example, with a validation of the region by simulation and an optional plot (requires matplotlib). |
| `examples/example2_global.py` | The global example, with a simulation from large initial states. |
| `examples/example3_pytorch.py` | Training of a PyTorch controller without biases, followed by its verification. |

The test suite (`pytest`) validates the results by simulation: for 25
randomly generated systems, every certified result is checked along
simulated solutions (nonnegativity of the state, invariance of the region,
and the guaranteed decay rate). When SciPy is installed, the vector computed
with `v="lp"` is also compared with the solution of SciPy's linear
programming solver.

## MATLAB

A MATLAB implementation with the same functions is provided in
[`matlab/`](matlab/README.md). It requires the MATLAB version of nnsector
and MATLAB R2016b or later, and it also runs in GNU Octave. Its results
agree with those of the Python implementation up to rounding errors.

## Citation

If you use nnstab in your research, please cite [1] and [2].

[1] H. Montazeri Hedesh, M. K. Wafi, and M. Siami, "Local Stability and
Region of Attraction Analysis for Neural Network Feedback Systems under
Positivity Constraints," in *Proc. 2025 IEEE 64th Conference on Decision
and Control (CDC)*, Rio de Janeiro, Brazil, 2025.
doi: [10.1109/CDC57313.2025.11312171](https://ieeexplore.ieee.org/document/11312171).
Preprint: [arXiv:2505.22889](https://arxiv.org/abs/2505.22889).

[2] H. Montazeri Hedesh and M. Siami, "Ensuring Both Positivity and
Stability Using Sector-Bounded Nonlinearity for Systems With Neural Network
Controllers," *IEEE Control Systems Letters*, vol. 8, pp. 1685-1690, 2024.
doi: [10.1109/LCSYS.2024.3416237](https://doi.org/10.1109/LCSYS.2024.3416237).
Preprint: [arXiv:2406.12744](https://arxiv.org/abs/2406.12744).

```bibtex
@inproceedings{montazerihedesh2025local,
  author    = {Montazeri Hedesh, Hamidreza and Wafi, Moh Kamalul and Siami, Milad},
  title     = {Local Stability and Region of Attraction Analysis for Neural Network
               Feedback Systems under Positivity Constraints},
  booktitle = {2025 IEEE 64th Conference on Decision and Control (CDC)},
  year      = {2025},
  doi       = {10.1109/CDC57313.2025.11312171}
}

@article{montazerihedesh2024ensuring,
  author  = {Montazeri Hedesh, Hamidreza and Siami, Milad},
  title   = {Ensuring Both Positivity and Stability Using Sector-Bounded Nonlinearity
             for Systems With Neural Network Controllers},
  journal = {IEEE Control Systems Letters},
  volume  = {8},
  pages   = {1685--1690},
  year    = {2024},
  doi     = {10.1109/LCSYS.2024.3416237}
}
```

The positive Aizerman result on which both tests rest is due to:

[3] R. Drummond, C. Guiver, and M. C. Turner, "Aizerman Conjectures for a
Class of Multivariate Positive Systems," *IEEE Transactions on Automatic
Control*, vol. 68, no. 8, pp. 5073-5080, 2023.
doi: [10.1109/TAC.2022.3217740](https://doi.org/10.1109/TAC.2022.3217740).

## License

nnstab is released under the BSD 3-Clause License. See [LICENSE](LICENSE).
