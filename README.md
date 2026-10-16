# nnstab

[![tests](https://github.com/hmon91/nnstab/actions/workflows/tests.yml/badge.svg)](https://github.com/hmon91/nnstab/actions/workflows/tests.yml)

**Stability checks for feedback loops with a neural network controller**, in
Python and MATLAB.

You give a plant and the trained network that closes the loop:

    x' = A x + B u,    y = C x,    u = pi(y)

nnstab checks whether the closed loop is **exponentially stable**: every
solution goes to 0 at least as fast as a decaying exponential. It works in
two steps:

1. **Global test.** Is the loop stable for every starting state `x0 >= 0`?
2. **Local test.** If the global test fails, nnstab looks for a region of
   starting states around 0 where the loop is stable, and makes it as large
   as it can.

Both tests use sector bounds of the network computed by
[nnsector](https://github.com/hmon91/nnsector).

## What the method needs

The tests are for *positive* systems, where the states never become negative:

* every entry of `B` and of `C` is `>= 0`,
* the starting states are `>= 0`,
* `pi(0) = 0`, so that `x = 0` is an equilibrium. A network without biases
  and with tanh, ReLU or leaky ReLU activations has this property.

`A` does not have to be stable on its own: the controller may be what makes
the loop stable. nnstab checks `B`, `C` and `pi(0)` and tells you if one of
them does not fit.

## Install

```bash
pip install git+https://github.com/hmon91/nnstab.git
```

This also installs nnsector. It needs Python 3.10 or newer and NumPy. For
PyTorch models, also install PyTorch (`pip install torch`). To work on the
code:

```bash
git clone https://github.com/hmon91/nnstab.git
cd nnstab
pip install -e ".[test]"
pytest
```

## Quick start

```python
import nnstab as nst

A, B, C, net = nst.example_system("local")     # or your own plant and network
res = nst.verify(A, B, C, net)
print(res)
```

```
Locally exponentially stable (the global test failed).
  region: {x >= 0 : v^T x <= 3.563},  v = [1.929, 2.089, 1]
  it holds every x0 >= 0 with sum(x0) <= 1.706
  in it: sum(x(t)) <= 2.09 exp(-0.000324 t) sum(x0)
  output box 0 <= y <= 1.848
  note: this is the largest region found, and near its edge the guaranteed decay rate is
  small; pass min_decay for a faster guaranteed rate on a smaller region.

Global check: not certified
  global sector slopes  Gamma1 = -4.791,  Gamma2 = 4.791
  A + B Gamma1 C Metzler: no,  A + B Gamma2 C Hurwitz: no
  A + B Gamma1 C is not Metzler: the global lower slope -Gamma2 is too negative to keep
  the states nonnegative.

Local check on 0 <= y <= 1.848: certified
  local sector slopes  gamma1 = -3.097,  gamma2 = -0.4766
  A + B gamma1 C Metzler: yes,  A + B gamma2 C Hurwitz: yes
  region: {x >= 0 : v^T x <= 3.563},  v = [1.929, 2.089, 1]
  it holds every x0 >= 0 with sum(x0) <= 1.706
  in it: sum(x(t)) <= 2.09 exp(-0.000324 t) sum(x0)
```

How to read it:

* **region**: every solution that starts in this set stays in it and goes
  to 0.
* **sum(x0) <= 1.706**: a simple way to see how big the region is. Every
  starting state with `x1 + x2 + x3 <= 1.706` is inside it.
* **exp(-0.000324 t)**: how fast the solutions go to 0, at least. Near the
  edge of the largest region this guaranteed rate is small. For a faster
  rate, ask for it, and you get a smaller region:

```python
res = nst.verify(A, B, C, net, min_decay=0.5)
# it holds every x0 >= 0 with sum(x0) <= 1.32
# in it: sum(x(t)) <= 2.1 exp(-0.501 t) sum(x0)
```

In code:

```python
res.status          # 'global', 'local' or 'none'
res.stable          # True if the global or the local test passed
res.contains(x0)    # True if the result covers the starting state x0
res.global_check    # details of the global test
res.local_check     # details of the local test (None if the global test passed)
```

### Your own system

```python
import numpy as np
import nnsector as ns
import nnstab as nst

A = np.array([[-3.0,  3.5,  2.0],
              [ 2.0, -4.0,  1.0],
              [ 3.5,  3.5, -5.0]])
B = np.array([[1.0], [0.5], [1.0]])
C = np.array([[1.0, 1.0, 0.5]])

# weights W_i have shape (l_i, l_{i-1}), like torch.nn.Linear.weight
net = ns.Network([W1, W2, W3], activation="tanh")    # no biases, so pi(0) = 0

res = nst.verify(A, B, C, net)
```

### From PyTorch

```python
model = torch.nn.Sequential(
    torch.nn.Linear(1, 16, bias=False), torch.nn.Tanh(),
    torch.nn.Linear(16, 16, bias=False), torch.nn.Tanh(),
    torch.nn.Linear(16, 1, bias=False),
)
# ... train the model ...
res = nst.verify(A, B, C, model)
```

`bias=False` keeps `pi(0) = 0`. The model is read with nnsector's
`Network.from_torch`.

## How it works

On a set of outputs, the network stays between two straight lines through
the origin:

    gamma1 y <= pi(y) <= gamma2 y

For several inputs or outputs, `gamma1` and `gamma2` are matrices and the
inequalities hold entry by entry. The positive Aizerman result [3] says: if
the two linear loops made with these slopes behave well, so does the loop
with the network.

**1. Global test.** nnsector gives slopes that hold for every `y >= 0`:
`Gamma1 = -Gamma2` and `Gamma2`. The loop is globally exponentially stable
for all `x0 >= 0` if

* `A + B Gamma1 C` is **Metzler**: all its entries off the diagonal are
  `>= 0`. This keeps the states from becoming negative.
* `A + B Gamma2 C` is **Hurwitz**: all its eigenvalues have a negative real
  part. This makes the states go to 0.

The global slopes are far apart, so this test is strict. It passes when the
network has a small effect on the loop.

**2. Local test.** On a box of outputs `0 <= y <= ybar`, nnsector gives
slopes `gamma1`, `gamma2` that are much closer together. If
`A + B gamma1 C` is Metzler and `A + B gamma2 C` is Hurwitz, nnstab finds a
vector `v > 0` with `v^T (A + B gamma2 C) < 0` and builds the region

    Omega = {x >= 0 : v^T x <= rho}

with `rho` as large as possible while every `x` in `Omega` has its output in
the box (`C x <= ybar`). In `Omega`, the sum `V(x) = v^T x` can only go down
along solutions. So a solution that starts in `Omega` stays in it, and

    v^T x(t) <= exp(-decay t) v^T x0.

**3. Search.** You do not have to pick the box. nnstab tries boxes
`0 <= y <= s * direction`: it doubles `s` (or halves it) until the test
changes its answer, finds the edge by bisection, and keeps the box whose
region holds the largest `sum(x0)`. A larger box gives slopes that are
farther apart, so at some size the test fails.

**The guarantee.** With `v_max` and `v_min` the largest and smallest
entries of `v`:

    sum(x(t)) <= (v_max / v_min) exp(-decay t) sum(x0)

The results hold `overshoot = v_max / v_min` and `decay`.

**The choice of v.** By default (`v="perron"`), `v` is the left eigenvector
of `A + B gamma2 C` for its largest eigenvalue. This gives the fastest
guaranteed decay. With `v="lp"`, nnstab picks the `v` that makes the region
as large as possible in every direction at once, while the decay rate stays
at least `min_decay` (or half the best rate, if `min_decay` is 0). This is a
linear program, but it has a special structure, so nnstab solves it exactly
with at most `n` linear solves and no LP solver.

**When a test fails.** These tests are *sufficient* conditions. When one
passes, the loop is stable. When both fail, the loop may still be stable;
the tests just cannot show it. The report says which condition failed.

## Options

`verify(A, B, C, net, ybar=None, direction=None, v="perron", min_decay=0.0, rtol=1e-3)`

| Option | What it does |
|---|---|
| `ybar` | Check only this output box `0 <= y <= ybar` (a number, or one number per output) instead of searching. |
| `direction` | The shape of the searched boxes `0 <= y <= s * direction`, when there are several outputs (default: all ones). |
| `v` | `"perron"` (default): the fastest guaranteed decay. `"lp"`: the largest region for a given decay rate. |
| `min_decay` | The smallest decay rate you accept (default 0). A larger value gives a smaller region. |
| `rtol` | How precisely the search finds the edge (default `1e-3`). |

## Functions

| Function | What it does |
|---|---|
| `verify(A, B, C, net, ...)` | The global test, then, if it fails, the search for a local region. Returns a `Verification`. |
| `check_global(A, B, C, net)` | Only the global test. Returns a `GlobalCheck`. |
| `check_local(A, B, C, net, ybar, v="perron", min_decay=0)` | Only the local test, on one box. Returns a `LocalCheck`. |
| `simulate(A, B, C, net, x0, t_end, dt=0.01)` | Simulates the loop (Runge-Kutta 4) from one starting state or many. |
| `sample_region(check, k, seed=0)` | `k` random starting states, spread evenly over a certified region. Good for testing a result by simulation. |
| `example_system(kind)` | The example loops `"local"` and `"global"`. Returns `A, B, C, net`. |
| `example_controller()` | The example network: 1 input, two hidden layers of 8 tanh neurons, 1 output, no biases. |
| `is_metzler`, `is_hurwitz`, `spectral_abscissa`, `hurwitz_certificate`, `perron_vector`, `copositive_vector`, `largest_region_vector` | The matrix tools behind the tests. |

The network can be an `nnsector.Network` or a `torch.nn.Sequential`. `B` may
be a vector for one input, and `C` a vector for one output.

What the results hold:

* `GlobalCheck`: `Gamma1`, `Gamma2`, `M1 = A + B Gamma1 C`,
  `M2 = A + B Gamma2 C`, `metzler`, `hurwitz`, `certified`, `decay`, `v`,
  `overshoot` and `reason` (why it failed).
* `LocalCheck`: the same with `gamma1` and `gamma2`, plus `ybar`, `rho`,
  `radius` (every `x0 >= 0` with `sum(x0) <= radius` is in the region),
  `method` and `contains(x)`.
* `Verification`: `status`, `stable`, `global_check`, `local_check`,
  `boxes_tried`, `note` and `contains(x)`.

`print()` gives a short report for each of them.

## The example loops

The example controller was trained to act like `u = -tanh(2y) - 0.5y` for
`0 <= y <= 3`. It has no biases, so `pi(0) = 0`. Near `y = 0` it pushes back
hard (`pi(y)` is about `-2.5 y`). For large `y` its output levels off, so it
pushes back more and more weakly.

* `example_system("local")`: the plant is unstable on its own (one
  eigenvalue of `A` is 0.95). Near 0 the controller makes the loop stable,
  but far away it is too weak: from `x0 = (4, 4, 4)` the state grows without
  bound. So no global test can pass. nnstab finds a region of attraction
  instead, which holds every `x0 >= 0` with `sum(x0) <= 1.706`.
* `example_system("global")`: a stable plant that the controller only
  nudges (small `B` and `C`). The global test passes.

## Examples and tests

* `examples/example1_local.py`: the local example. Checks the result by
  simulation from random starting states in the region, and plots them if
  matplotlib is installed.
* `examples/example2_global.py`: the global example, with a simulation from
  large starting states.
* `examples/example3_pytorch.py`: trains a small PyTorch controller without
  biases, then verifies the loop.
* `pytest` runs the tests. Among them: for 25 random loops, every result is
  checked by simulation (the states stay `>= 0`, stay in the region, and go
  to 0 at least at the guaranteed rate), and the region vector of
  `v="lp"` is compared with SciPy's LP solver when SciPy is installed.

## MATLAB

The same library for MATLAB (and GNU Octave) is in
[`matlab/`](matlab/README.md), with the same functions. It needs the MATLAB
version of nnsector.

## Citing

If you use nnstab, please cite:

[1] H. Montazeri Hedesh, M. K. Wafi, and M. Siami, "Local Stability and
Region of Attraction Analysis for Neural Network Feedback Systems under
Positivity Constraints," in *2025 IEEE 64th Conference on Decision and
Control (CDC)*, Rio de Janeiro, Brazil, 2025.
[IEEE Xplore](https://ieeexplore.ieee.org/document/11312171),
[arXiv:2505.22889](https://arxiv.org/abs/2505.22889)

[2] H. Montazeri Hedesh and M. Siami, "Ensuring Both Positivity and
Stability Using Sector-Bounded Nonlinearity for Systems With Neural Network
Controllers," *IEEE Control Systems Letters*, vol. 8, pp. 1685-1690, 2024.
[doi:10.1109/LCSYS.2024.3416237](https://doi.org/10.1109/LCSYS.2024.3416237),
[arXiv:2406.12744](https://arxiv.org/abs/2406.12744)

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

The positive Aizerman result that both tests rest on:

[3] R. Drummond, C. Guiver, and M. C. Turner, "Aizerman Conjectures for a
Class of Multivariate Positive Systems," *IEEE Transactions on Automatic
Control*, vol. 68, no. 8, pp. 5073-5080, 2023.
[doi:10.1109/TAC.2022.3217740](https://doi.org/10.1109/TAC.2022.3217740)

## License

BSD 3-Clause, see [LICENSE](LICENSE).
