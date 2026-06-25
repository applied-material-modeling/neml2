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

(tutorials-extension-input-files)=
# Connecting to input files

You'll take a `Model` subclass you wrote yourself and make it
loadable from an HIT input file, the same way the built-in models in
[](tutorials-models-running-your-first-model) are. Two ingredients do
the work: a decorator that registers the class under a type name, and
the `HitSchema` you already saw in
[](tutorials-extension-arguments).

```{code-cell} ipython3
:tags: [remove-cell]

# When this notebook runs in Google Colab, install NEML2 from PyPI. The guard
# makes the cell a no-op everywhere else (the docs build and local Jupyter
# already have NEML2 installed), and the cell is hidden from the rendered docs.
import sys

if "google.colab" in sys.modules:
    !pip install -q neml2
```

## The example model

We'll reuse the projectile model from the previous tutorial — a `Vec`
velocity in, a `Vec` acceleration out, under gravity plus linear drag:

$$
\boldsymbol{a} \;=\; \boldsymbol{g} \;-\; \mu\, \boldsymbol{v},
$$

with $\mu$ a `Scalar` parameter and $\boldsymbol{g}$ a constant `Vec`
buffer. Write it out as `projectile.py`:

```{code-cell} ipython3
%%writefile projectile.py
# A minimal NEML2 Model subclass illustrating the input-file wiring.
#
# Forward operator: gravity-plus-linear-drag projectile acceleration,
#     a = g - mu * v
# with `g` a Vec buffer (constant gravity) and `mu` a single Scalar
# parameter. The point of this file is the *connection to HIT* — the
# schema declaration and the `@register_neml2_object` decorator — not the
# physics.

from __future__ import annotations

from neml2.factory import register_neml2_object
from neml2.models.model import Model
from neml2.schema import HitSchema, buffer, input, output, parameter
from neml2.types import Scalar, Vec


@register_neml2_object("ProjectileAcceleration")
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

Two things to notice:

1. **`@register_neml2_object("ProjectileAcceleration")`** is what
   makes `type = ProjectileAcceleration` work in an input file. The
   string is the HIT type name; by convention it matches the Python
   class name, but it doesn't have to.
2. **`hit = HitSchema(...)`** lists the options the input block
   accepts. Each `input` / `output` / `parameter` / `buffer` entry
   becomes one HIT option of the same name. See
   [](tutorials-extension-arguments) for the full set of field
   helpers and what they accept.

## The input file

Here's a minimal HIT block that instantiates the class:

```{code-cell} ipython3
%%writefile input.i
[Models]
  [accel]
    type = ProjectileAcceleration
    velocity = 'v'
    acceleration = 'a'
    dynamic_viscosity = 0.05
  []
[]
```

Each line maps to one schema field:

- `type = ProjectileAcceleration` picks the class out of the registry.
- `velocity = 'v'` and `acceleration = 'a'` rename the input/output
  variables — without these lines the canonical names (`velocity`,
  `acceleration`) are used.
- `dynamic_viscosity = 0.05` sets the parameter value. It lands on
  the instance as `self.mu` (the attribute we asked for via
  `attr="mu"`).

## Loading and inspecting

`load_model` looks up the type name in a registry that's populated
as a side effect of importing your module — so you need to import
it first. In a normal script that's just an `import` line; here,
`projectile.py` sits next to `input.i`, so we add the directory to
`sys.path` and import it.

```{code-cell} ipython3
import sys

sys.path.insert(0, ".")
import projectile  # the @register_neml2_object runs at import time

import neml2

model = neml2.load_model("input.i", "accel")
model
```

The renames from the input file show up in `input_spec` and
`output_spec`:

```{code-cell} ipython3
model.input_spec, model.output_spec
```

The canonical names (`velocity`, `acceleration`) were replaced by the
HIT values (`v`, `a`). These are the names sibling models in a
`ComposedModel` would use to wire to ours — see
[](tutorials-extension-composition).

The parameter is exposed under the `attr` name we picked:

```{code-cell} ipython3
model.mu
```

## Debugging "not registered" errors

If `load_model` raises a `KeyError` mentioning *"not registered in
NativeRegistry"*, the module holding `@register_neml2_object` was
almost certainly never imported. Fix the import (or reorder imports)
so the decorator runs first. To confirm a type is registered, ask
`neml2-syntax`:

```bash
neml2-syntax --load ./projectile.py --type ProjectileAcceleration
```

## Loading the extension from the CLI

To use your custom model from the `neml2-*` shell tools, pass
`--load` pointing at the file (or a dotted module name on
`sys.path`). The flag runs the import before the input file is
parsed, so the registered type is ready by the time `load_model` is
called:

```bash
neml2-run --load ./projectile.py input.i driver
neml2-inspect --load ./projectile.py input.i accel
```

Repeat `--load` for multiple extensions — they import in the given
order, so later modules can depend on names registered earlier. See
[](cli-utilities) for the full list of tools.

## Where to go next

- [](tutorials-extension-arguments) covers the schema fields in
  detail — typed inputs, options, parameters that can be promoted to
  runtime inputs.
- [](tutorials-extension-forward) is the next step: fill in
  `forward()` so the model actually computes something.
- Once your model loads, you'll usually want to compose it with
  others inside a single `[Models]` block — see
  [](tutorials-extension-composition).
