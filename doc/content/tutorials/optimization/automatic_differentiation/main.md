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

(tutorials-optimization-autograd)=
# Automatic differentiation

Calibrating a material model means finding parameter values that make
the model's predictions match observations. The gradient of a scalar
loss with respect to the parameters is what every gradient-based
optimizer (SGD, Adam, L-BFGS, …) needs to take a step. NEML2 piggy-backs
on PyTorch's autograd engine to provide that gradient for free: every
`Model` is a `torch.nn.Module`, every trainable parameter is a
`torch.nn.Parameter`, and every tensor operation inside the model is
traced.

This page walks through the mechanics: enable the gradient tape, run
the model, build a scalar loss, call `.backward()`, and read the
per-parameter gradient off `parameter.grad`.

## The model

We reuse the linear-isotropic-elastic model from
[](tutorials-models-running-your-first-model) so the focus stays on
autograd rather than the physics:

```{literalinclude} input.i
:language: ini
:caption: input.i
```

```{code-cell} ipython3
import neml2

model = neml2.load_model("input.i", "elasticity")
model
```

## Parameters are `nn.Parameter`s

Because `Model` derives from `torch.nn.Module`, the model's named
parameters show up under `model.named_parameters()` exactly like any
other PyTorch module:

```{code-cell} ipython3
for name, p in model.named_parameters():
    print(f"{name:>4} = {p.item():<10g}  requires_grad={p.requires_grad}")
```

`model.E` and `model.nu` are NEML2 `Scalar` wrappers around those
parameters; the underlying `torch.nn.Parameter` lives at `model.E.data`
and `model.nu.data`. Because parameters default to `requires_grad=True`,
**no extra opt-in is needed** — every forward pass through the model
already records the autograd graph.

:::{note}
If you want to *freeze* a parameter (treat it as a constant during
calibration), set `requires_grad=False` on its `.data`:

```python
model.nu.data.requires_grad_(False)
```
:::

## A scalar loss, end-to-end

Pick a small uniaxial-strain input and evaluate the model:

```{code-cell} ipython3
import torch
from neml2.types import SR2

strain = SR2.fill(0.01, 0.0, 0.0, 0.0, 0.0, 0.0)
stress = model(strain)
stress
```

Notice the `grad_fn=…` annotation on the underlying tensor — the
forward call wired up the autograd graph that connects `stress` back to
`model.E` and `model.nu`. Any scalar function of `stress` can now be
back-propagated. As a stand-in for a real calibration objective, drop
down to the underlying `torch.Tensor` via `stress.data` so we can
call PyTorch's tensor `.norm()` directly — autograd is preserved
through this access:

```{code-cell} ipython3
loss = stress.data.norm()
loss
```

```{code-cell} ipython3
loss.backward()
print(f"dL/dE  = {model.E.data.grad.item():.6g}")
print(f"dL/dnu = {model.nu.data.grad.item():.6g}")
```

That is the entire autograd workflow:

1. Load the model (parameters already track gradients).
2. Evaluate the model on some input.
3. Combine the output(s) into a scalar `loss`.
4. Call `loss.backward()`.
5. Read each parameter's gradient off `parameter.data.grad`.

## Per-component derivatives via `torch.autograd.grad`

`.backward()` accumulates into `.grad`, which is convenient inside a
training loop but inconvenient when you want a one-off derivative
without side effects. `torch.autograd.grad` returns the gradient
directly:

```{code-cell} ipython3
strain = SR2.fill(0.01, 0.0, 0.0, 0.0, 0.0, 0.0)
stress = model(strain)

# Differentiate the xx-component of stress w.r.t. both parameters.
sigma_xx = stress.data[0]
dE, dnu = torch.autograd.grad(sigma_xx, [model.E.data, model.nu.data])
print(f"d(sigma_xx)/dE  = {dE.item():.6g}")
print(f"d(sigma_xx)/dnu = {dnu.item():.6g}")
```

For a uniaxial-strain test the closed-form prediction is
$\sigma_{xx} = (1-\nu)\,E\,\varepsilon_{xx}/[(1+\nu)(1-2\nu)]$, which
gives $\partial \sigma_{xx}/\partial E = (1-\nu)\varepsilon_{xx}/[(1+\nu)(1-2\nu)] \approx 0.01346$
at $E=200$ GPa, $\nu=0.3$, $\varepsilon_{xx}=0.01$ — matching the
autograd value to machine precision.

:::{warning}
Each call to `.backward()` **adds to** `.grad`. In a training loop you
must zero the gradients between optimizer steps, either explicitly
(`model.E.data.grad = None`) or via the optimizer's
`zero_grad()` method. Forgetting to do so leads to silently wrong
updates.
:::

## What about implicit models?

For an implicit model — one whose output comes from an internal Newton
solve, e.g. a viscoplastic update — autograd still flows back to the
parameters with no changes to your training code. The reason is that
`ImplicitUpdate` overrides the backward rule: at the converged
solution it applies the implicit function theorem so the gradient
comes from a single linear solve against the Jacobian at the fixed
point, instead of recording every intermediate iterate from every
step in the autograd tape. The result is the analytically exact
derivative at the converged point, and the backward pass costs one
linear solve regardless of how many Newton iterations the forward
pass took.

You don't have to do anything special to take advantage of this — as
long as the implicit residual is wrapped in `ImplicitUpdate`, autograd
on the model output flows straight back to the parameters just like in
the explicit case above. See
[](tutorials-models-implicit-model) for how implicit models are
assembled in NEML2.

## Where to go next

- [](tutorials-optimization-calibration) puts these gradients to work
  in a full calibration loop with a PyTorch optimizer.
- For recurrent (time-stepped) calibration where the loss depends on
  an entire load history, see the `pyzag` adapter at
  [](tutorials-optimization-pyzag).
