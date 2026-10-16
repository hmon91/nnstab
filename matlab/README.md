# nnstab for MATLAB

Stability checks for feedback loops with a neural network controller, in
MATLAB and GNU Octave. The Python package with the same method is described
in the [main README](../README.md).

You give a plant and the trained network that closes the loop:

    x' = A x + B u,    y = C x,    u = pi(y)

nnstab checks whether the closed loop is exponentially stable. First it
tries the **global test** (stable for every starting state `x0 >= 0`). If
that fails, it looks for a **local region** of starting states around 0
where the loop is stable, and makes it as large as it can. Both tests use
sector bounds of the network computed by
[nnsector](https://github.com/hmon91/nnsector).

The method needs `B >= 0`, `C >= 0` (every entry), starting states
`x0 >= 0`, and `pi(0) = 0` (for example, a network without biases with
tanh, ReLU or leaky ReLU activations).

## Setup

nnstab needs the MATLAB version of nnsector. Put both on the path:

```matlab
addpath('path/to/nnsector/matlab')     % the folder that contains +nnsector
addpath('path/to/nnstab/matlab')       % the folder that contains +nnstab
```

Needs MATLAB R2016b or newer. It also runs in GNU Octave (tested with 8.4).

## Quick start

```matlab
[A, B, C, net] = nnstab.example_system('local');   % or your own plant and network
res = nnstab.verify(A, B, C, net);
nnstab.report(res)
```

```
Locally exponentially stable (the global test failed).
  region: {x >= 0 : v'*x <= 3.563},  v = [1.929, 2.089, 1]
  it holds every x0 >= 0 with sum(x0) <= 1.706
  in it: sum(x(t)) <= 2.09 exp(-0.000324 t) sum(x0)
  output box 0 <= y <= 1.848
  note: this is the largest region found, and near its edge the guaranteed decay rate is
  small; pass MinDecay for a faster guaranteed rate on a smaller region.
...
```

* **region**: every solution that starts in this set stays in it and goes
  to 0.
* **sum(x0) <= 1.706**: every starting state with `x1 + x2 + x3 <= 1.706`
  is inside the region.
* **exp(-0.000324 t)**: how fast the solutions go to 0, at least. For a
  faster rate on a smaller region:

```matlab
res = nnstab.verify(A, B, C, net, 'MinDecay', 0.5);
% it holds every x0 >= 0 with sum(x0) <= 1.32
% in it: sum(x(t)) <= 2.1 exp(-0.501 t) sum(x0)
```

In code:

```matlab
res.status                    % 'global', 'local' or 'none'
res.stable                    % true if the global or the local test passed
nnstab.in_region(res, x0)     % true if the result covers the starting state x0
res.global_check              % details of the global test
res.local_check               % details of the local test ([] if the global test passed)
```

### Your own system

```matlab
A = [-3.0  3.5  2.0
      2.0 -4.0  1.0
      3.5  3.5 -5.0];
B = [1.0; 0.5; 1.0];
C = [1.0 1.0 0.5];
net = nnsector.network({W1, W2, W3});    % W{i} is l_i x l_{i-1}; no biases, so pi(0) = 0
res = nnstab.verify(A, B, C, net);
```

## How it works

On a set of outputs, the network stays between two straight lines through
the origin: `gamma1*y <= pi(y) <= gamma2*y` (entry by entry for several
inputs or outputs).

1. **Global test.** nnsector gives `Gamma2` for every `y >= 0`, with
   `Gamma1 = -Gamma2`. The loop is globally exponentially stable for all
   `x0 >= 0` if `A + B*Gamma1*C` is Metzler (all entries off the diagonal
   are `>= 0`, so the states stay `>= 0`) and `A + B*Gamma2*C` is Hurwitz
   (all eigenvalues have a negative real part, so the states go to 0).
2. **Local test.** On a box of outputs `0 <= y <= ybar`, nnsector gives
   slopes `gamma1`, `gamma2` that are much closer together. If
   `A + B*gamma1*C` is Metzler and `A + B*gamma2*C` is Hurwitz, nnstab finds
   `v > 0` with `v'*(A + B*gamma2*C) < 0`. Then the region
   `{x >= 0 : v'*x <= rho}` is a region of attraction, with `rho` as large
   as possible while `C*x <= ybar` on it. Along solutions in it,
   `v'*x(t) <= exp(-decay*t) * v'*x0`.
3. **Search.** nnstab tries boxes `0 <= y <= s*d`: it doubles `s` (or
   halves it) until the test changes its answer, finds the edge by
   bisection, and keeps the box whose region holds the largest `sum(x0)`.

The tests are sufficient conditions: when one passes, the loop is stable;
when both fail, the loop may still be stable. The main README explains the
method in more detail.

## Functions

| Function | What it does |
|---|---|
| `nnstab.verify(A, B, C, net, ...)` | The global test, then, if it fails, the search for a local region. Options: `'ybar'` (check only this box), `'Direction'` (shape of the searched boxes, default all ones), `'v'`, `'MinDecay'` and `'RelTol'` (default `1e-3`). |
| `nnstab.check_global(A, B, C, net)` | Only the global test. |
| `nnstab.check_local(A, B, C, net, ybar, ...)` | Only the local test, on one box. Options: `'v'` (`'perron'`, the default, for the fastest guaranteed decay, or `'lp'` for the largest region with a given decay rate) and `'MinDecay'` (default 0). |
| `nnstab.report(r)` | Prints a short report of any result (or returns it as text). |
| `nnstab.in_region(r, X)` | True if a result covers every starting state (column) of `X`. |
| `nnstab.simulate(A, B, C, net, x0, t_end, dt)` | Simulates the loop (Runge-Kutta 4, default `dt = 0.01`) from one starting state or many (one per column). |
| `nnstab.sample_region(chk, k, seed)` | `k` random starting states, spread evenly over a certified region. |
| `nnstab.example_system(kind)` | The example loops `'local'` and `'global'`. Returns `[A, B, C, net]`. |
| `nnstab.example_controller()` | The example network: 1 input, two hidden layers of 8 tanh neurons, 1 output, no biases. |
| `nnstab.check_system(A, B, C, net)` | Checks the sizes, `B >= 0`, `C >= 0` and `pi(0) = 0`. |
| `nnstab.is_metzler`, `nnstab.is_hurwitz`, `nnstab.spectral_abscissa`, `nnstab.hurwitz_certificate`, `nnstab.perron_vector`, `nnstab.copositive_vector`, `nnstab.largest_region_vector` | The matrix tools behind the tests. |

What the results hold:

* `check_global`: `Gamma1`, `Gamma2`, `M1 = A + B*Gamma1*C`,
  `M2 = A + B*Gamma2*C`, `metzler`, `hurwitz`, `certified`, `decay`, `v`,
  `overshoot` and `reason` (why it failed). If certified, every solution
  with `x0 >= 0` has `sum(x(t)) <= overshoot * exp(-decay*t) * sum(x0)`.
* `check_local`: the same with `gamma1` and `gamma2`, plus `ybar`, `rho`,
  `radius` (every `x0 >= 0` with `sum(x0) <= radius` is in the region) and
  `method`.
* `verify`: `status`, `stable`, `global_check`, `local_check`,
  `boxes_tried` and `note`.

For one input, `B` may be a row or a column; for one output, so may `C`.

## Examples and tests

* `examples/example1.m`: the local example. Checks the result by
  simulation from random starting states in the region, and plots them.
* `examples/example2.m`: the global example, with a simulation from large
  starting states.
* `tests/run_tests.m`: from the `tests` folder, run `run_tests`. Some tests
  compare with the numbers of the Python package.

The examples and the tests find nnsector through the environment variable
`NNSECTOR_MATLAB` (the folder that contains `+nnsector`), or in a folder
`nnsector` next to the `nnstab` folder.
