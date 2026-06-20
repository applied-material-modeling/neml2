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

This tutorial walks through how to declare the inputs, outputs, and
parameters of a custom model. The next tutorial in the series
([](tutorials-extension-forward)) shows how to fill in the actual
computation.

## The running example

A projectile of velocity $\boldsymbol{v}$ in a viscous medium feels
gravity plus linear drag:

$$
  \boldsymbol{a} \;=\; \boldsymbol{g} \;-\; \mu \, \boldsymbol{v}.
$$

So the model needs:

- an input — the velocity $\boldsymbol{v}$ (a `Vec`),
- an output — the acceleration $\boldsymbol{a}$ (also a `Vec`),
- a parameter — the drag coefficient $\mu$ (a `Scalar` we may want to
  fit to data later, so it has to be calibratable), and
- a buffer — the gravitational acceleration $\boldsymbol{g}$ (a known
  constant of the environment).

## The schema

A model declares its inputs, outputs, parameters, and buffers in one
class-level `hit = HitSchema(...)` block. Each line is a single call
to a helper from `neml2.schema`:

```{code-cell} ipython3
from __future__ import annotations

from neml2.factory import register_neml2_object
from neml2.models.model import Model
from neml2.schema import HitSchema, buffer, input, output, parameter
from neml2.types import Scalar, Vec


class ProjectileAcceleration(Model):
    """Projectile acceleration under gravity and linear drag: a = g - mu * v."""

    hit = HitSchema(
        input("velocity", Vec, "Projectile velocity"),
        output("acceleration", Vec, "Projectile acceleration"),
        parameter("dynamic_viscosity", Scalar, "Drag coefficient mu", attr="mu"),
        buffer(
            "gravity",
            Vec,
            "Gravitational acceleration vector",
            attr="g",
            default=Vec.fill(0.0, -9.81, 0.0),
        ),
    )

    def forward(self, velocity, *nl_params, v=None):
        raise NotImplementedError
```

A few notes on the schema:

- The string in each declaration (`"velocity"`, `"dynamic_viscosity"`,
  …) is the name the input file will use.
- The third argument is the field's docstring — it shows up in the
  auto-generated syntax catalog.
- `attr="mu"` on the parameter exposes it on the instance as
  `self.mu`, so the model body can write `self.mu * velocity`
  directly.
- A `buffer` is a constant that travels with the model (e.g., goes to
  GPU when the model does) but isn't trainable.

## Variable declaration

```python
input("velocity", Vec, "Velocity of the projectile")
```

`input(name, type_cls, doc)` declares one input variable. `type_cls`
is one of the typed wrappers from `neml2.types` — `Scalar`, `Vec`,
`R2`, `SR2`, `Rot`, etc. — and tells the framework what shape and
algebra to expect. `output(...)` works the same way; it declares an
output instead.

The name string serves two purposes: it's both the HIT option name
the user writes in the input file, AND the default variable name when
no rename is supplied. So a fresh instance reports
`input_spec = {"velocity": Vec}`:

```{code-cell} ipython3
m = ProjectileAcceleration(dynamic_viscosity=0.001)
m.input_spec
```

```{code-cell} ipython3
m.output_spec
```

If the user passes a rename, the new name shows up in the spec dict
instead:

```{code-cell} ipython3
m_renamed = ProjectileAcceleration(
    velocity="v",
    acceleration="a",
    dynamic_viscosity=0.001,
)
m_renamed.input_spec, m_renamed.output_spec
```

That's what lets the same class slot into many different input files
— each file picks the names it wants.

:::{note}
The order of `input(...)` calls matters: it determines the order of
positional arguments to `forward()`. Same for `output(...)` and the
return tuple. The next tutorial,
[](tutorials-extension-forward), shows how that works.
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

A parameter declaration takes a name, a type, a docstring, and the
attribute it should be exposed under (`mu` here). Other useful options:

- `default=...` makes the parameter optional in the input file — if
  the user omits it, the schema default is used.
- `allow_nonlinear=True` lets the parameter be promoted to a runtime
  input — useful when you want to drive the parameter from another
  model's output. See [](tutorials-extension-composition) for the
  details.

`attr="mu"` just renames the storage slot so the model body can write
`self.mu` instead of `self.dynamic_viscosity` — handy when the math
uses Greek-letter conventions. The underlying storage is a
`torch.nn.Parameter` (visible in `named_parameters()`), so the usual
PyTorch training idioms work unchanged. Attribute access via `self.mu`
returns a typed `Scalar` wrapper that carries the same data:

```{code-cell} ipython3
dict(m.named_parameters())
```

```{code-cell} ipython3
m.mu
```

The model is a `torch.nn.Module`, so the usual PyTorch idioms work
unchanged — `model.to(device)`, `model.state_dict()`,
`torch.optim.Adam(model.parameters(), ...)`, etc.

## Inspecting the declared surface

Once the declarations are in place, the surface is available as
attributes:

```{code-cell} ipython3
print("input_spec :", m.input_spec)
print("output_spec:", m.output_spec)
print("parameters :", [name for name, _ in m.named_parameters()])
print("buffers    :", [name for name, _ in m.named_buffers()])
```

That's everything you need to declare the *shape* of a model. The
next two tutorials cover the rest:

- [](tutorials-extension-input-files) — how this class shows up in
  a HIT input file so users construct it without touching Python.
- [](tutorials-extension-forward) — the `forward()` body that turns
  these declarations into actual computation.
