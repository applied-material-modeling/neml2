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

The previous tutorial declared the projectile model's inputs, outputs,
and parameters. This one fills in the math: a `forward()` method that
turns a velocity into an acceleration.

## The `forward` method

`forward()` is the function NEML2 calls to evaluate your model. The
positional arguments are the inputs in the order you declared them in
the schema — each one is already a typed wrapper (`Vec`, `SR2`, …) the
framework built for you. Return the outputs in schema order.

A `v=None` keyword argument is also part of the signature. Ignore it
for now — the pure-forward path is `v is None`, which is what nearly
every caller hits. We'll come back to `v` further down when we add
first derivatives.

## Implementation

Recall the equation for the projectile model:

$$
  \boldsymbol{a} = \boldsymbol{g} - \mu \boldsymbol{v}.
$$

`self.g` is the gravity vector, `self.mu` is the drag coefficient, and
the velocity comes in as the first positional argument. The whole
forward is:

```{literalinclude} projectile.py
:language: python
:caption: projectile.py
:pyobject: ProjectileAcceleration.forward
```

The first line is the physics: `Vec - Scalar * Vec` gives back a `Vec`,
batched or not. If `v is None` (the usual case) the method returns and
you're done.

The `else` branch is the chain-rule hook. `actions` maps each input to
a closure that computes its contribution to the Jacobian-vector
product. Here $\partial \boldsymbol{a}/\partial \boldsymbol{v} = -\mu
I$, so the closure is just `lambda V: -self.mu * V`.
`apply_chain_rule` does the rest. The closure is matrix-free — nothing
ever materializes a Jacobian block.

## Evaluation

That's the whole model. Load it the same way you'd load any built-in
type — `neml2.load_model` finds it through the factory as long as the
module that registers it has been imported.

The input file from the previous tutorial wires the model into HIT:

```{literalinclude} input.i
:language: ini
:caption: input.i
```

Import the module so the class registers, then load and call:

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

Because `forward()` implements the `v` branch, the same model can hand
back directional derivatives with no extra wiring. Seed a tangent on
the velocity input and the JVP comes back through `v_out`:

```{code-cell} ipython3
# Seed the identity on Vec to read off the full Jacobian column ∂a/∂v.
seed = Vec(torch.eye(3, dtype=torch.float64))
accel, v_out = model(vel, v={"v": {"velocity_leaf": seed}})
v_out["a"]["velocity_leaf"].data
```

This matches the analytical $\partial \boldsymbol{a}/\partial
\boldsymbol{v} = -\mu I = -0.001\, I_3$ exactly.

## Driving the model from a unit-test input

`ModelUnitTest` is the usual way to pin a custom model's behavior in
CI. It loads the model, runs it on inputs you supply, checks the
outputs, and cross-checks the derivatives against PyTorch's autograd:

```{literalinclude} unit_test.i
:language: ini
:caption: unit_test.i
```

```{code-cell} ipython3
from neml2.drivers.ModelUnitTest import ModelUnitTest

report = ModelUnitTest.from_file("unit_test.i").run()
print(f"value checks: {report.value_checks}, JVP checks: {report.jvp_checks}")
```

Both passes succeed — values match, derivatives match autograd.

## Where to go next

- The next tutorial, [](tutorials-extension-composition), shows how to
  glue several models together so a dependency resolver wires their
  inputs and outputs and threads the chain rule for you.
- For richer chain-rule examples (multiple inputs, non-linear leaves)
  the source of
  {class}`~neml2.models.common.LinearCombination._LinearCombination` and
  {class}`~neml2.models.solid_mechanics.plasticity.YieldFunction.YieldFunction`
  is a good read.
