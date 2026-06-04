---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
mystnb:
  execution_mode: cache
---

(tutorials-models-transient-driver)=
# Transient driver

The tutorials so far have evaluated a model at a single state: feed in
some forces, get back some state. Real material simulations want the
*trajectory* — strain ramped from zero to a maximum over many steps, with
each step's converged solution feeding the next. NEML2 packages this
recursive update inside a `[Drivers]` block called
[`TransientDriver`](drivers-TransientDriver), built and run from Python via the
same `load_input` factory introduced in the
[](tutorials-models-input-file) tutorial.

## The recursive update

A single-step constitutive update has the form

$$
  \mathbf{s}_{n+1}
  = f(\mathbf{f}_{n+1},\, \mathbf{s}_n,\, \mathbf{f}_n;\, \mathbf{p}),
$$

where $\mathbf{s}_n$ are the state variables at step $n$,
$\mathbf{f}_n$ the driving forces, and $\mathbf{p}$ the parameters. The
[](tutorials-models-implicit-model) tutorial shows how `ImplicitUpdate`
wraps a residual into exactly this map.

What that map does *not* describe is the trajectory itself: given an
initial state $\mathbf{s}_0$ and a prescribed force history
$\{\mathbf{f}_n\}_{n=0}^{N}$, you still have to thread converged outputs
back in as the next step's inputs. `TransientDriver` is the loop that
does it:

```python
for n in range(N):
    s[n + 1] = model(f[n + 1], s[n], f[n])
```

When coupling NEML2 to an external PDE solver, that loop usually lives
outside NEML2 — the host solver supplies $\mathbf{f}_{n+1}$ and asks
NEML2 for $\mathbf{s}_{n+1}$. For self-contained workflows
(verification, regression tests, parameter calibration, plotting a
stress–strain curve), `TransientDriver` keeps everything inside NEML2.

:::{note}
For training and adjoint-style sensitivities through a transient,
prefer [`pyzag`](https://github.com/applied-material-modeling/pyzag),
which performs the same recursive update in a much more efficient fashion.
:::

## The input file

A `[Drivers]` block names the model to step, the prescribed time
history, and the prescribed driving-force histories (here a single
symmetric strain tensor `E`). Time and forces are pulled from named
`[Tensors]` blocks:

```{literalinclude} input.i
:language: ini
:caption: input.i
```

A few things worth pointing out:

- `prescribed_time` is a `Scalar` of shape `(N,)` — one time value per
  step. `N` sets the number of steps the driver will take.
- `force_<Type>_names` / `force_<Type>_values` are the prescribed
  driving forces. The trailing `_names` lists *variable* names that the
  model exposes; the matching `_values` lists names of `[Tensors]`
  blocks that supply their step-by-step values. Each force tensor must
  carry a leading axis of length `N`.
- Anything not listed as a prescribed force or initial condition starts
  at zero. For this perfect-viscoplastic model, that means stress is
  zero at step 0.
- The `model` referenced by the driver is the same kind of object you
  would call directly from Python — a single-step forward operator. The
  driver just wraps the time loop around it.

The `[Tensors]` block uses NEML2's `Python` tensor type to build the
prescribed arrays inline: a 50-step time vector $t \in [0, 1]$ and a
matching strain history that linearly ramps to a peak uniaxial strain
of $\varepsilon_{xx} = 5\%$ with lateral contractions consistent with
isochoric deformation. See [](tutorials-models-input-file) for the full
`type = Python` syntax.

## Building and running the driver

`load_input(path).get_driver(name)` builds the driver from the input
file:

```{code-cell} ipython3
import neml2

factory = neml2.load_input("input.i")
driver = factory.get_driver("driver")
driver
```

The driver carries the parsed model, the prescribed-time tensor, and
the prescribed forces. The number of steps comes from the leading axis
of `prescribed_time`:

```{code-cell} ipython3
print("nsteps        =", driver.nsteps)
print("forces        =", list(driver.forces))
print("wrapped model =", type(driver.model).__name__)
```

`driver.run()` executes the time loop. Each step calls the model once,
threading the previous step's converged state in as the history input
for the next step. The method returns `True` on success:

```{code-cell} ipython3
driver.run()
```

## Reading the per-step output

`driver.result()` returns a flat dict keyed by
`input.<step>.<variable>` and `output.<step>.<variable>`, with raw
`torch.Tensor` values. Step 0 holds only the prescribed forces (the
model is not called at step 0); every later step holds whatever the
forward operator returned:

```{code-cell} ipython3
results = driver.result()
print("total entries:", len(results))
print("step 0  keys:", [k for k in results if k.startswith("input.0.")])
print("step 1  inputs:", [k for k in results if k.startswith("input.1.")])
print("step 1  outputs:", [k for k in results if k.startswith("output.1.")])
```

The `~k` suffix on a variable name denotes the value from `k` steps
back — this is how `TransientDriver` exposes history dependence to the
forward operator. `E~1` is the previous step's strain, `t~1` the
previous step's time. The single-step model sees both `E` and `E~1`
and computes a strain rate internally.

Pulling per-step strain and stress values out as 1-D tensors is a
simple comprehension:

```{code-cell} ipython3
import torch

nsteps = driver.nsteps
times = torch.tensor([results[f"input.{i}.t"].item() for i in range(nsteps)])
strain_xx = torch.tensor(
    [results[f"input.{i}.E"][0].item() for i in range(nsteps)]
)
# Step 0 has no output -> stress is the zero initial condition.
stress_xx = torch.tensor(
    [0.0] + [results[f"output.{i}.stress"][0].item() for i in range(1, nsteps)]
)
print(f"max strain_xx = {strain_xx.max().item():.4f}")
print(f"max stress_xx = {stress_xx.max().item():.2f}")
```

(Index `[0]` picks the first Mandel slot of the `SR2` payload, which is
the $xx$ component.)

## Plotting the stress–strain curve

With per-step `strain_xx` and `stress_xx` in hand, the curve is just a
`matplotlib` call. `matplotlib` is available in the `[dev]` extras
([](tutorials-models-input-file) installs it):

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(strain_xx, stress_xx, "C0o-", markersize=3)
ax.set_xlabel(r"$\varepsilon_{xx}$")
ax.set_ylabel(r"$\sigma_{xx}$")
ax.grid(True, alpha=0.3)
fig.tight_layout()
```

The initial linear segment is purely elastic. Once the von Mises stress
exceeds the yield surface the Perzyna flow rule activates and the
stress plateaus at the rate-dependent overstress level — a textbook
perfect-viscoplastic response.

## Saving the trajectory to disk

The full result dict can be persisted to a single `.pt` file. The
layout is a flat `{key: tensor}` dict keyed by the same
`input.<step>.<var>` / `output.<step>.<var>` strings that
[`driver.result()`](drivers-TransientDriver) returns in memory:

```{code-cell} ipython3
driver.save_gold("result.pt")

loaded = torch.load("result.pt", weights_only=True)
print("entry count:", len(loaded))
print("a few keys :", list(loaded)[:4])
```

This is the same format used by the regression suite's
[`TransientRegression`](drivers-TransientRegression) — the gold file
checked into `tests/regression/.../gold/result.pt` is exactly what
`save_gold` produces. Reading it back from another script is just
`torch.load(path, weights_only=True)`.

## Where to go next

- The driver above prescribes strain. To prescribe stress instead, swap
  the unknowns and residuals of the equation system and change the
  `force_*` block to supply the stress history; the rest of the
  workflow is unchanged.
- Every step is a batched call. Give `prescribed_time` and the force
  tensors a *trailing* batch axis and the driver will sweep that axis
  in parallel — useful for studying parameter or initial-condition
  sensitivities. See [](tutorials-models-vectorization) for the
  underlying batching rules.
- For gradient-based training through a transient response, see
  [`pyzag`](https://github.com/applied-material-modeling/pyzag), which
  implements the same recursive update under autograd.
