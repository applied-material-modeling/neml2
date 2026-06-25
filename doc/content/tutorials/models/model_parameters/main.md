---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(tutorials-models-parameters)=
# Model parameters

You'll pick up the `LinearIsotropicElasticity` model from
[](tutorials-models-running-your-first-model) and read, change, and
differentiate through its moduli from Python.

```{code-cell} ipython3
:tags: [remove-cell]

# When this notebook runs in Google Colab, install NEML2 from PyPI. The guard
# makes the cell a no-op everywhere else (the docs build and local Jupyter
# already have NEML2 installed), and the cell is hidden from the rendered docs.
import sys

if "google.colab" in sys.modules:
    !pip install -q neml2
```

## The input file

Same input file as the previous tutorial — re-used here so we can focus
on the parameters:

```{code-cell} ipython3
%%writefile input.i
# Same linear-isotropic-elasticity model as the previous tutorial.
# Re-used here so the focus stays on parameter access / mutation rather
# than on a new physical setup.
[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients      = '200e3          0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
```

## Listing the parameters

A NEML2 model is a `torch.nn.Module`, so `named_parameters()` works
the same way it does for any PyTorch model:

```{code-cell} ipython3
import neml2

model = neml2.load_model("input.i", "elasticity")

for name, param in model.named_parameters():
    print(f"{name}: {param.data.item()}")
```

The two parameters `E` and `nu` correspond to the two `coefficients`
we passed in the input file.

## Reading a parameter by name

Each parameter is also exposed as a Python attribute on the model.
Reading `model.E` returns a `Scalar` — one of NEML2's typed tensor
wrappers (see [](tensor-types) for the full list):

```{code-cell} ipython3
print("E  =", model.E)
print("nu =", model.nu)
```

The underlying `torch.nn.Parameter` lives at `.data`. Use `.data`
when you need the raw `torch.Tensor` — for example to pull out a
numerical value with `.item()`. (This is user-facing; library code
stays on the typed wrappers.)

```{code-cell} ipython3
print("E  =", model.E.data.item())
print("nu =", model.nu.data.item())
```

## Changing a parameter

There are two ways to change a parameter at runtime. Pick based on
whether you want to keep the existing `nn.Parameter` (and any
optimizer state attached to it) or replace it outright.

### In place, on `.data`

In-place mutation keeps the same `nn.Parameter` object. Leaf tensors
that require grad can't be mutated in place directly — autograd would
lose the history. Wrap the mutation in `torch.no_grad()` to tell
PyTorch you're intentionally side-stepping the graph for this
assignment:

```{code-cell} ipython3
import torch

with torch.no_grad():
    model.E.data.fill_(150e3)

print("E =", model.E.data.item())
```

Use this inside an optimization loop — the optimizer holds a
reference to the underlying `nn.Parameter` (the one
`named_parameters()` returns), and in-place mutation through `.data`
writes back to that same Parameter.

### Rebinding the attribute

Assigning a fresh `torch.nn.Parameter` swaps the slot wholesale:

```{code-cell} ipython3
model.nu = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
print("nu =", model.nu.data.item())
```

Reach for this when the new parameter has different properties — a
different dtype, a different `requires_grad`, or a batched shape
(see [](tutorials-models-parameters-revisited) for batched parameter
values, or [](tutorials-models-vectorization) for batched inputs).
Rebinding invalidates any optimizer that was tracking the old
`nn.Parameter`, so re-create the optimizer afterwards if there was
one.

## Freezing a parameter

Toggle `requires_grad` off and the parameter is excluded from
autodiff — its gradient stays `None` through a backward pass, and any
optimizer tracking it will leave it alone:

```{code-cell} ipython3
model.E.data.requires_grad_(False)
print("E.requires_grad =", model.E.data.requires_grad)
```

Use this to hold one modulus fixed while calibrating the other —
see [](tutorials-optimization-calibration) for the full workflow.

## Parameters flow into outputs

Changing a parameter changes the next evaluation's output. Reload the
model so the state is clean, then evaluate it once, halve `E`, and
evaluate again:

```{code-cell} ipython3
from neml2.types import SR2

model = neml2.load_model("input.i", "elasticity")
strain = SR2.fill(0.01, 0.0, 0.0, 0.0, 0.0, 0.0)

print("E = 200e3 ->", model(strain).data)

model.E = torch.nn.Parameter(torch.tensor(100e3, dtype=torch.float64))
print("E = 100e3 ->", model(strain).data)
```

For linear elasticity the stress scales linearly with $E$, so the
second row is exactly half the first. Any parameter change shows up on
the next forward call — no need to re-load the input file.

## Differentiating through parameters

Autodiff flows through NEML2 parameters just like it does through any
other `nn.Module` weight. Backward from a scalar reduction of the
output gives gradients on each parameter:

```{code-cell} ipython3
model = neml2.load_model("input.i", "elasticity")
strain = SR2.fill(0.01, 0.0, 0.0, 0.0, 0.0, 0.0)

stress = model(strain)
stress.data.sum().backward()

print("dL/dE  =", model.E.data.grad.item())
print("dL/dnu =", model.nu.data.grad.item())
```

This is what calibration workflows build on — embed the model in a
larger PyTorch graph, define a loss against experimental data, and
optimize the moduli with any `torch.optim` optimizer.
[](tutorials-optimization-autograd) and
[](tutorials-optimization-calibration) walk the full pipeline; for
calibration through a time-stepped transient, see
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
