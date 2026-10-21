# nnstab for MATLAB

This folder contains the MATLAB implementation of nnstab, which verifies the
exponential stability of positive feedback systems with a neural network
controller. It is also compatible with GNU Octave. The problem statement,
the assumptions and the method are the same as in the Python package and are
described in the [main README](../README.md).

In brief, nnstab considers the closed loop `x' = A x + B u`, `y = C x`,
`u = pi(y)` with `B >= 0`, `C >= 0` and `pi(0) = 0`. It first tests global
exponential stability, for all initial states `x0 >= 0`, with the global
sector bound of the network. If this test fails, it computes the largest
certified region of attraction it can find with local sector bounds. The
sector bounds are computed with [nnsector](https://github.com/hmon91/nnsector).

## Setup

nnstab requires the MATLAB version of nnsector. Both libraries must be on
the path:

```matlab
addpath('path/to/nnsector/matlab')     % the folder that contains +nnsector
addpath('path/to/nnstab/matlab')       % the folder that contains +nnstab
```

The library requires MATLAB R2016b or later. It also runs in GNU Octave
(tested with version 8.4).

## Quick start

```matlab
[A, B, C, net] = nnstab.example_system('local');   % or a user-defined system
res = nnstab.verify(A, B, C, net);
nnstab.report(res)
```

The first lines of the output are:

```
Locally exponentially stable (the global test failed).
  region: {x >= 0 : v'*x <= 3.563},  v = [1.929, 2.089, 1]
  it holds every x0 >= 0 with sum(x0) <= 1.706
  in it: sum(x(t)) <= 2.09 exp(-0.000324 t) sum(x0)
  output box 0 <= y <= 1.848
```

The meaning of these lines is explained in the
[main README](../README.md#quick-start). A larger guaranteed decay rate is
obtained on a smaller region with the option `'MinDecay'`:

```matlab
res = nnstab.verify(A, B, C, net, 'MinDecay', 0.5);
% The region contains every x0 >= 0 with sum(x0) <= 1.32,
% and sum(x(t)) <= 2.1 exp(-0.501 t) sum(x0) holds on it.
```

The result can also be used in code:

```matlab
res.status                    % 'global', 'local' or 'none'
res.stable                    % true if the global or the local test succeeded
nnstab.in_region(res, x0)     % true if the result covers the initial state x0
res.global_check              % details of the global test
res.local_check               % details of the local test ([] after a global result)
```

**User-defined systems.**

```matlab
A = [-3.0  3.5  2.0
      2.0 -4.0  1.0
      3.5  3.5 -5.0];
B = [1.0; 0.5; 1.0];
C = [1.0 1.0 0.5];
net = nnsector.network({W1, W2, W3});   % W{i} is l_i x l_{i-1}; no biases, hence pi(0) = 0
res = nnstab.verify(A, B, C, net);
```

## Functions

| Function | Description |
|---|---|
| `nnstab.verify(A, B, C, net, ...)` | Global test, followed by the search for a local region if the global test fails. Options: `'ybar'` (test only this box), `'Direction'` (direction of the searched boxes, default all ones), `'v'`, `'MinDecay'` and `'RelTol'` (default `1e-3`). |
| `nnstab.check_global(A, B, C, net)` | Global test only. |
| `nnstab.check_local(A, B, C, net, ybar, ...)` | Local test on one box. Options: `'v'` (`'perron'` (default) for the largest guaranteed decay rate, or `'lp'` for the largest region with a prescribed decay rate) and `'MinDecay'` (default 0). |
| `nnstab.report(r)` | Prints a short report of any result, or returns it as text. |
| `nnstab.in_region(r, X)` | True if a result covers every initial state (column) of `X`. |
| `nnstab.simulate(A, B, C, net, x0, t_end, dt)` | Simulates the closed loop (fourth-order Runge-Kutta method, default `dt = 0.01`) from one or several initial states (one per column). |
| `nnstab.sample_region(chk, k, seed)` | Draws `k` initial states uniformly from a certified region. |
| `nnstab.example_system(kind)` | The example systems `'local'` and `'global'`. Returns `[A, B, C, net]`. |
| `nnstab.example_controller()` | The example network: one input, two hidden layers of eight tanh neurons, one output, and no biases. |
| `nnstab.check_system(A, B, C, net)` | Checks the dimensions and the assumptions `B >= 0`, `C >= 0` and `pi(0) = 0`. |
| `nnstab.is_metzler`, `nnstab.is_hurwitz`, `nnstab.spectral_abscissa`, `nnstab.hurwitz_certificate`, `nnstab.perron_vector`, `nnstab.copositive_vector`, `nnstab.largest_region_vector` | Matrix routines used by the tests. |

For one input, `B` may be given as a row or a column; for one output, the
same holds for `C`.

## Results

* `nnstab.check_global` returns a structure with the fields `Gamma1`,
  `Gamma2`, `M1` (the matrix `A + B*Gamma1*C`), `M2` (the matrix
  `A + B*Gamma2*C`), `metzler`, `hurwitz`, `certified`, `decay`, `v`,
  `overshoot` and `reason` (the cause of a failure). If the test succeeds,
  every solution with `x0 >= 0` satisfies
  `sum(x(t)) <= overshoot * exp(-decay*t) * sum(x0)`.
* `nnstab.check_local` returns the same fields with `gamma1` and `gamma2`,
  together with `ybar`, `rho`, `radius` (every `x0 >= 0` with
  `sum(x0) <= radius` lies in the region) and `method`. The region is
  `{x >= 0 : v'*x <= rho}`.
* `nnstab.verify` returns a structure with the fields `status`, `stable`,
  `global_check`, `local_check`, `boxes_tried` and `note`.

## Examples and tests

| File | Content |
|---|---|
| `examples/example1.m` | The local example, with a validation of the region by simulation and a plot. |
| `examples/example2.m` | The global example, with a simulation from large initial states. |
| `tests/run_tests.m` | The test suite. Run `run_tests` from the `tests` folder. Several tests compare the results with those of the Python package. |

The examples and the tests locate nnsector through the environment variable
`NNSECTOR_MATLAB` (the folder that contains `+nnsector`) or in a folder
named `nnsector` next to the `nnstab` folder.

## Citation

See the [main README](../README.md#citation).
