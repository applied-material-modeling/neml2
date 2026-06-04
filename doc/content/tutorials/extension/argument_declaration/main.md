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

(tutorials-extension-arguments)=
# Declaring inputs, outputs, and parameters

Every NEML2 model — whether it ships with the library or you wrote it
yourself — can be written in the form

$$
  y = f(x;\, p, b),
$$

where

- $x$ are the *input variables* the model consumes,
- $y$ are the *output variables* it produces,
- $p$ are *parameters* (trainable, calibration-tracked tensors that
  travel with the model and participate in autograd), and
- $b$ are *buffers* (non-trainable tensors that also travel with the
  model — sent to GPU when the model is, baked into the AOTI graph
  when exported).

Writing your own model therefore breaks down into four steps:

1. Declare the input variables $x$.
2. Declare the output variables $y$.
3. Declare the parameters $p$ and the buffers $b$.
4. Define the forward operator $f$.

This tutorial covers the first three — the *declarative surface* of a
`Model` subclass. The forward operator is the subject of
[](tutorials-extension-forward).

## The running example

The example we'll carry through this series is a viscous-projectile
acceleration model. A projectile of velocity $\boldsymbol{v}$ in a
viscous medium experiences

$$
  \boldsymbol{a} \;=\; \boldsymbol{g} \;-\; \mu \, \boldsymbol{v},
$$

where $\boldsymbol{g}$ is the gravitational acceleration and $\mu$ is
the dynamic viscosity of the medium. So the model has

- one input variable $\boldsymbol{v}$ (`Vec`),
- one output variable $\boldsymbol{a}$ (`Vec`),
- one *parameter* $\mu$ (we want to calibrate it from data, so it has
  to be a trainable scalar), and
- one *buffer* $\boldsymbol{g}$ (a known constant of the environment;
  we don't want autograd flowing through it).

## The schema

Native NEML2 models declare their HIT-facing surface with a
class-level `hit = HitSchema(...)` block. `HitSchema` is the
Python-native counterpart of the C++ `expected_options()` static
method, but smaller and declarative: each field is one call to a
helper from `neml2.schema`.

The minimum needed for our example:

```{code-cell} ipython3
from __future__ import annotations

import torch

from neml2.model import Model
from neml2.schema import HitSchema, input, output, parameter
from neml2.types import Vec, Scalar


class ProjectileAcceleration(Model):
    """Acceleration of a projectile in a viscous medium."""

    hit = HitSchema(
        input("velocity", Vec, "Velocity of the projectile"),
        output("acceleration", Vec, "Acceleration of the projectile"),
        parameter(
            "dynamic_viscosity",
            Scalar,
            "Dynamic viscosity of the medium",
            attr="mu",
        ),
    )

    def __post_init__(self) -> None:
        # Gravitational acceleration is a non-trainable constant — register
        # it as a typed buffer in __post_init__, after the schema has
        # populated parameters but before forward() can run.
        g = Vec.fill(0.0, -9.81, 0.0)
        self.register_typed_buffer("g", g)

    def forward(self, velocity, *nl_params, v=None):
        # Forward operator deferred to the next tutorial.
        raise NotImplementedError
```

A few things are happening above:

- `hit = HitSchema(...)` is a *class-level* attribute. The base class's
  `__init_subclass__` reads it and derives the class-level
  `input_spec` / `output_spec` dicts automatically — no boilerplate.
- The strings `"velocity"`, `"acceleration"`, `"dynamic_viscosity"`
  are the *HIT option names* the input file will use. They double as
  the canonical variable names when no override is supplied.
- The third positional argument on every field — `"Velocity of the
  projectile"`, etc. — is the docstring. It flows into the
  auto-generated syntax catalog (`neml2-syntax`); empty docstrings
  are rejected at class-definition time.
- `attr="mu"` on the parameter field tells the framework to expose the
  registered parameter on the instance as `self.mu`, so the forward
  operator can write `self.mu * velocity` without going through a
  string lookup.

## Variable declaration

```python
input("velocity", Vec, "Velocity of the projectile")
```

`input(name, type_cls, doc)` declares one input variable. The
`type_cls` is one of the typed wrappers from `neml2.types`
(`Scalar`, `Vec`, `R2`, `SR2`, `Rot`, `Quaternion`, `SSR4`, …); it
both documents the tensor shape (a `Vec` has base shape `(3,)`, an
`SR2` has base shape `(6,)` in Mandel packing, etc.) and gates the
runtime tensor wrapping inside the framework.

`output(name, type_cls, doc)` is exactly symmetric — same signature,
declares an output variable instead.

The string `"velocity"` plays two roles:

1. It is the HIT option name. When the model is loaded from an input
   file, the parser looks for `velocity = '...'` under the model
   block to discover what *external* name the input variable goes by
   inside the wider computation graph.
2. It is the *default* canonical variable name. If the user doesn't
   supply `velocity = '...'`, the input is exposed under the literal
   name `"velocity"`.

That's why a fresh instance reports `input_spec = {"velocity": Vec}`:

```{code-cell} ipython3
m = ProjectileAcceleration(dynamic_viscosity=0.001)
m.input_spec
```

```{code-cell} ipython3
m.output_spec
```

When the user *does* supply a rename, the resolved name shows up in
the spec dict instead — both for direct Python construction and for
HIT-file loading:

```{code-cell} ipython3
m_renamed = ProjectileAcceleration(
    velocity="v",
    acceleration="a",
    dynamic_viscosity=0.001,
)
m_renamed.input_spec, m_renamed.output_spec
```

This is what lets the same `ProjectileAcceleration` class slot into
many different input files: each file picks the external variable
names it wants, and the model self-rebuilds its `input_spec` /
`output_spec` to match.

:::{note}
The order of `input(...)` calls in the schema **is** significant: it
determines the order of positional arguments to `forward()`. The
order of `output(...)` calls is likewise the order of the
return-tuple. The next tutorial,
[](tutorials-extension-forward), walks through this.
:::

## Parameter declaration

```python
parameter(
    "dynamic_viscosity",
    Scalar,
    "Dynamic viscosity of the medium",
    attr="mu",
)
```

`parameter(name, type_cls, doc, *, attr=None, default=..., allow_nonlinear=False)`
declares one trainable scalar/tensor parameter:

- `name` is the HIT option the input file uses to supply the value.
- `type_cls` is the typed wrapper the parameter is wrapped in.
- `doc` is the syntax-catalog docstring (same ASCII/required-string
  rule as variables).
- `attr` is the attribute the registered `torch.nn.Parameter` is
  exposed under. The forward operator should reach for the value via
  `self._get_param("mu", nl_params, Scalar)` rather than
  `self.mu` directly — that helper transparently handles the case
  where the parameter has been promoted to a runtime input (see
  `allow_nonlinear` below).
- `default` (optional) makes the parameter optional in the input
  file; if omitted there, the schema default is used instead.
- `allow_nonlinear=True` opts in to *parameter promotion* — the user
  can name another model's output (or a bare variable) in the
  parameter slot, and the framework will silently turn that
  parameter into an additional runtime input. See
  [](tutorials-extension-composition) for the composition story.

Because `attr="mu"` is set, the parameter is now a real
`torch.nn.Parameter` registered on the module, surfaces in
`named_parameters()`, and participates in autograd:

```{code-cell} ipython3
dict(m.named_parameters())
```

```{code-cell} ipython3
m.mu
```

```{code-cell} ipython3
# It's a leaf tensor with requires_grad=True, so a loss built from it
# can backprop into a calibration loop.
m.mu.data.requires_grad
```

The model is a `torch.nn.Module`, so the usual PyTorch idioms work
unchanged — `model.to(device)`, `model.state_dict()`,
`torch.optim.Adam(model.parameters(), ...)`, etc.

## Buffer declaration

There is **no** dedicated `buffer(...)` schema helper, because
buffers are typically *constants* of the model — environment values,
lookup tables, geometric data — that the user shouldn't have to
re-supply via HIT every time. The idiomatic place to register them
is the model's `__post_init__` hook, using
{meth}`Model.register_typed_buffer`:

```python
def __post_init__(self) -> None:
    g = Vec.fill(0.0, -9.81, 0.0)
    self.register_typed_buffer("g", g)
```

`__post_init__` runs at the end of `Model.__init__`, after the schema
has populated any `attr`-declared options on `self` but *before* the
forward operator can be called. The registered buffer:

- appears in `named_buffers()`,
- moves with the module under `.to(device)`,
- is baked as a constant during AOTI export, and
- does **not** participate in autograd.

```{code-cell} ipython3
dict(m.named_buffers())
```

```{code-cell} ipython3
m.g
```

If the constant *does* depend on a HIT option (say, the user picks
the gravity vector per-problem), declare an `option(...)` field for
that knob and read it from `self.<attr>` inside `__post_init__` before
calling `register_typed_buffer`. The `neml2.schema.option` helper
documented in the schema module supports `str`, `int`, `float`, and
`bool` HIT values out of the box.

## Inspecting the declared surface

Once all the declarations are in place, the *structure* of the model
is fully determined: the framework knows what tensors flow in, what
tensors flow out, what parameters can be calibrated, and what
buffers ride along. A quick `repr` already summarizes it:

```{code-cell} ipython3
m
```

For programmatic introspection, all four surfaces are first-class
attributes:

```{code-cell} ipython3
print("input_spec :", m.input_spec)
print("output_spec:", m.output_spec)
print("parameters :", [name for name, _ in m.named_parameters()])
print("buffers    :", [name for name, _ in m.named_buffers()])
```

That's the whole declarative surface. The next tutorial,
[](tutorials-extension-input-files), shows how this same class is
wired into a HIT input file so a user can construct it without
touching Python at all; after that,
[](tutorials-extension-forward) fills in the `forward()` body that
turns these declarations into actual computation.
