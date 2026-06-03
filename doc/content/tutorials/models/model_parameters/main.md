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

(tutorials-models-parameters)=
# Model parameters

The previous tutorial,
[](tutorials-models-running-your-first-model), loaded a
`LinearIsotropicElasticity` model with hard-coded Young's modulus and
Poisson's ratio. This tutorial picks up the same model and shows how
NEML2 exposes those moduli — and the surface for reading and mutating
them from Python.

## Inputs, outputs, parameters, and buffers

Every NEML2 model is a map of the form

$$
  y = f(x;\, p, b),
$$

with four distinct kinds of data:

- **Input variables** ($x$) — the model's per-evaluation inputs. They
  are passed as positional arguments to the model call and change on
  every step. For the elasticity model in `input.i`, $x$ is a single
  symmetric strain tensor (`SR2`).
- **Output variables** ($y$) — what the model returns. For elasticity,
  $y$ is the symmetric stress tensor (`SR2`).
- **Parameters** ($p$) — the *trainable* coefficients of the model.
  They live on the `torch.nn.Module` as `nn.Parameter` slots, so they
  participate in autodiff and can be optimized — see the
  [](tutorials-optimization) tutorials for the calibration workflow.
  For elasticity, $p$ is $\{E, \nu\}$.
- **Buffers** ($b$) — non-trainable persistent state that travels with
  the module (registered as `nn.Buffer`). Buffers move with the module
  when you call `.to(device)`, get saved into `state_dict()`, etc., but
  they do **not** receive gradients.

## The input file

The input file is identical to the previous tutorial — re-used here so
we can focus on the parameter surface rather than introducing a new
physics setup:

```{literalinclude} input.i
:language: ini
:caption: input.i
```

## Enumerating parameters

A loaded NEML2 model *is* a `torch.nn.Module`, so the standard PyTorch
introspection works. `named_parameters()` yields every registered
parameter with its hierarchical name:

```{code-cell} ipython3
import neml2

model = neml2.load_model("input.i", "elasticity")

for name, param in model.named_parameters():
    print(f"{name}: {param.data.item()}")
```

`LinearIsotropicElasticity` registered two parameters here, `E` and
`nu`, matching the two `coefficients` we passed in. No buffers are
registered — `list(model.named_buffers())` is empty for this model.

## Reading a parameter by name

Each registered parameter is also exposed as a Python attribute on the
model. Reading `model.E` returns the typed wrapper (`Scalar`, here)
that NEML2 uses for the parameter's tensor type:

```{code-cell} ipython3
print("E  =", model.E)
print("nu =", model.nu)
```

The underlying `torch.nn.Parameter` lives at `.data`, which is what
you need when you want the raw `torch.Tensor` — for example to read a
numerical value:

```{code-cell} ipython3
print("E  =", model.E.data.item())
print("nu =", model.nu.data.item())
```

## Mutating a parameter

There are two equivalent ways to change a parameter at runtime. Which
one to pick depends on whether you want to keep the existing
`nn.Parameter` (and any optimizer state attached to it) or replace it
outright.

### In place, on `.data`

In-place mutation of `model.E.data` keeps the same `nn.Parameter`
object. PyTorch refuses in-place writes to a leaf tensor that requires
grad, so wrap the mutation in `torch.no_grad()`:

```{code-cell} ipython3
import torch

with torch.no_grad():
    model.E.data.fill_(150e3)

print("E =", model.E.data.item())
```

This is the right tool for parameter updates inside an optimization
loop — the optimizer already holds a reference to `model.E`, and
in-place mutation leaves that reference valid.

### Rebinding the attribute

Assigning a fresh `torch.nn.Parameter` to the attribute swaps the
slot wholesale:

```{code-cell} ipython3
model.nu = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
print("nu =", model.nu.data.item())
```

This is the right tool when you want a parameter with different
properties (different dtype, different `requires_grad`, different
shape for a vectorized batch — covered in
[](tutorials-models-vectorization)). It invalidates any optimizer
that was tracking the old `nn.Parameter`, so re-create the optimizer
afterwards if there was one.

## Freezing parameters with `requires_grad`

Every `nn.Parameter` carries a `requires_grad` flag. Toggling it off
on a parameter excludes it from autodiff — the gradient stays `None`
through a backward pass, and any optimizer over that parameter will
leave it alone:

```{code-cell} ipython3
model.E.data.requires_grad_(False)
print("E.requires_grad =", model.E.data.requires_grad)
```

Use this to hold one modulus fixed while calibrating the other —
see [](tutorials-optimization-calibration) for the full workflow.

## Parameters flow into outputs

Mutating a parameter changes the next evaluation's output. Reload the
model so the state is clean, then evaluate it once, halve `E`, and
evaluate again:

```{code-cell} ipython3
from neml2.types import SR2

model = neml2.load_model("input.i", "elasticity")
strain = SR2(torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))

print("E = 200e3 ->", model(strain).data)

model.E = torch.nn.Parameter(torch.tensor(100e3, dtype=torch.float64))
print("E = 100e3 ->", model(strain).data)
```

For linear elasticity the stress scales linearly with $E$, so the
second row is exactly half the first. The point is more general: any
parameter change shows up on the next forward call without re-loading
the input file.

## Differentiating through parameters

Because parameters are leaf tensors with `requires_grad=True`, autodiff
flows through them just like through any other `nn.Module` weight.
Backward from a scalar reduction of the output gives gradients on each
parameter:

```{code-cell} ipython3
model = neml2.load_model("input.i", "elasticity")
strain = SR2(torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))

stress = model(strain)
stress.data.sum().backward()

print("dL/dE  =", model.E.data.grad.item())
print("dL/dnu =", model.nu.data.grad.item())
```

This is the foundation NEML2 builds on for calibration workflows —
embed a NEML2 model inside a larger PyTorch graph, define a loss
against experimental data, and optimize the moduli with any
`torch.optim` optimizer. [](tutorials-optimization-autograd) and
[](tutorials-optimization-calibration) work the full pipeline; for
recurrent (time-stepped) calibration through a transient see
[](tutorials-optimization-pyzag).

## Where to go next

- A parameter doesn't have to be a literal — it can also be supplied
  by another model, promoted to an input on the host, or shared
  across siblings in a `ComposedModel`. See
  [](tutorials-models-parameters-revisited).
- Real material models compose several pieces (elasticity, hardening,
  flow rule, …), each contributing its own parameters. The mechanism
  is the subject of [](tutorials-models-cross-referencing) and
  [](tutorials-models-composition).
- Batched evaluation — both over the inputs and over the parameters
  themselves — is covered in [](tutorials-models-vectorization).
- Once you're comfortable reading and mutating parameters, the
  [](tutorials-optimization) tutorials cover calibrating them
  against experimental data via PyTorch autograd.
