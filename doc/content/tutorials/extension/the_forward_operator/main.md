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

(tutorials-extension-forward)=
# The forward operator

The previous two tutorials wired up the *structure* of the projectile
model — its inputs, outputs, parameters, and the HIT block that
instantiates it. What's left is the math: the body of `forward()` that
turns a velocity into an acceleration, optionally accompanied by the
chain-rule machinery for first-order derivatives through composed models.

## Method signature

A Python-native NEML2 model derives from
{class}`neml2.model.Model` and implements one method:

```python
def forward(self, *typed_inputs, v=None):
    ...
```

The positional arguments are the structural inputs in the order they
appear in `input_spec` — each one is already a typed wrapper
(`Scalar`, `Vec`, `SR2`, …) the framework constructed for you. The
single keyword argument `v` carries the first-order tangent seeded on
each leaf input:

| kwarg | type | role |
|---|---|---|
| `v` | `dict[str, dict[str, TensorWrapper]]` \| `None` | First-order tangent seeded on each leaf input. |

`v=None` is the pure-forward case and is what almost every caller
uses. When the caller does seed `v`, the forward additionally returns
the JVP (Jacobian-vector product) of every output against every input
tangent, propagated through the model's local Jacobian.

:::{note}
For the specialised case of leaves that sit inside a
{class}`~neml2.models.solid_mechanics.plasticity.Normality` wrap (which
needs the second-order chain rule), the signature gains two extra
opt-in kwargs `v2` / `vh` and the leaf has to declare
`SUPPORTS_SECOND_ORDER = True`. That path is rarely needed by user
models — see {class}`~neml2.models.common.SR2Invariant.SR2Invariant` for
a worked example if you do.
:::

## Implementation

Recall the equation for the projectile model:

$$
  \boldsymbol{a} = \boldsymbol{g} - \mu \boldsymbol{v}.
$$

With the schema declared in [](tutorials-extension-arguments), `self.g`
is a `Vec` (the gravity parameter), `self.mu` is a `Scalar` (the
dynamic viscosity), and the velocity arrives as the first positional
argument. The forward method is:

```{literalinclude} projectile.py
:language: python
:caption: projectile.py
:pyobject: ProjectileAcceleration.forward
```

Three things are happening:

1. **Compute the value.** The arithmetic is plain typed-wrapper
   algebra: `Vec - Scalar * Vec` returns a `Vec` of the correct shape,
   batched or not. This branch alone is enough for the pure-forward
   contract — if `v is None`, the method returns the output and stops.
2. **Declare the local Jacobian.** `actions` is a dict keyed by *input
   variable name* (the user-facing name resolved from HIT, not the
   schema option name). Each value is a closure that receives an
   incoming tangent `V` for that input and returns the contribution
   to ∂(output)/∂(seed-leaf). Here $\partial \boldsymbol{a}/\partial
   \boldsymbol{v} = -\mu I$, so the closure is `lambda V: -self.mu * V`
   — the framework handles the matrix-free pushforward.
3. **Dispatch through `apply_chain_rule`.**
   {meth}`~neml2.model.Model.apply_chain_rule` consumes the incoming
   `v` channel together with the `actions` dict and returns the JVP
   keyed by output and seed leaf. The forward returns the tuple
   `(value, v_out)`.

:::{note}
The closure `lambda V: -self.mu * V` is matrix-free: the framework
hands it whatever shape of incoming tangent the caller seeded, and the
returned typed wrapper is accumulated by seed leaf. Nothing in the
forward ever materializes a Jacobian block.
:::

## Evaluation

The model definition is now *complete*: structural declarations from
the schema, registration with the factory, and a body for `forward()`.
The custom model can be evaluated through the same `neml2.load_model`
entry point as any built-in type.

The input file from the previous tutorial wires the model into HIT:

```{literalinclude} input.i
:language: ini
:caption: input.i
```

Import the module that contains the registered class, then load and
call the model.

```{code-cell} ipython3
import sys, os
sys.path.insert(0, os.getcwd())

import projectile  # registers ProjectileAcceleration with the native factory
import neml2

model = neml2.load_model("input.i", "accel")
model
```

```{code-cell} ipython3
import torch
from neml2.types import Vec

vel = Vec.fill(10.0, 2.0, 0.0)
accel = model(vel)
accel
```

The result is $\boldsymbol{g} - \mu \boldsymbol{v} = (0, -9.81, 0) -
0.001 \cdot (10, 2, 0) = (-0.01, -9.812, 0)$ — the typed wrapper
preserves the `Vec` shape on the way out.

## First derivatives

Because the forward implements the `v` channel, the same model can also
report directional derivatives without any extra wiring. Seed an
incoming tangent on the velocity input and the JVP comes back keyed by
output and by seed leaf:

```{code-cell} ipython3
# A K=3 batch of seed tangents — the identity on Vec — yields the full
# Jacobian column of ∂a/∂v evaluated against I. ``v`` is keyed
# {input_var: {seed_leaf: tangent}}; ``v_out`` is keyed the same way
# but with the output variable on the outer level.
seed = Vec(torch.eye(3, dtype=torch.float64))
accel, v_out = model(vel, v={"v": {"velocity_leaf": seed}})
v_out["a"]["velocity_leaf"].data
```

This matches the analytical $\partial \boldsymbol{a}/\partial
\boldsymbol{v} = -\mu I = -0.001\, I_3$ exactly.

## Driving the model from a unit-test input

The `ModelUnitTest` driver is the convention for pinning a custom
leaf's behavior in CI: it loads the model from HIT, runs the forward
on user-supplied inputs, checks values against
`output_<Type>_values`, and cross-checks the first-derivative
implementation against PyTorch's autograd JVP oracle. The driver
discovers the model automatically once the module that registers it
has been imported.

```{literalinclude} unit_test.i
:language: ini
:caption: unit_test.i
```

```{code-cell} ipython3
from neml2.drivers.ModelUnitTest import ModelUnitTest

report = ModelUnitTest.from_file("unit_test.i").run()
print(f"value checks: {report.value_checks}, JVP checks: {report.jvp_checks}")
```

Both passes — values against the analytical answer and the analytical
JVP against autograd — succeed, confirming the forward body is
internally consistent.

## What's next

- The model so far is a *single leaf*. The next tutorial,
  [](tutorials-extension-composition), shows how to glue several
  leaves into a `ComposedModel`, where each leaf's `forward` is called
  by the dependency resolver and the chain-rule contributions are
  threaded automatically.
- For richer chain-rule examples — multiple inputs, non-linear
  leaves — read the source of
  {class}`~neml2.models.common.LinearCombination._LinearCombination`,
  {class}`~neml2.models.solid_mechanics.plasticity.YieldFunction.YieldFunction`,
  and
  {class}`~neml2.models.common.ConstantParameter._ConstantParameter`.
