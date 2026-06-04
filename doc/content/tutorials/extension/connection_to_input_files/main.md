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

(tutorials-extension-input-files)=
# Connecting to input files

The first tutorial in this series showed how to call a built-in model
from an input file. The job of this page is to wire a *new* model
class — one you wrote yourself — into the same input-file machinery,
so that an HIT block of the form

```ini
[Models]
  [accel]
    type = ProjectileAcceleration
    ...
  []
[]
```

instantiates your class. The mechanism is a single decorator plus a
small declarative schema, both living next to the `Model` subclass.

## The registry-factory pattern

NEML2 uses a classic [registry-factory
pattern](https://en.wikipedia.org/wiki/Factory_method_pattern) for
runtime object creation:

1. Every `Model` subclass registers itself under a string *type name*
   when its module is imported.
2. `neml2.load_model(path, name)` parses the HIT file, reads the
   `type = ...` field of the requested block, looks the type name up
   in the registry, and asks the matched class to build itself from
   the HIT node.

There are three pieces a model author touches:

`@register_native("TypeName")`
: Decorator on the class. Adds the class to the native registry
  under the given HIT type name. Lives in {mod}`neml2.factory`.

`HitSchema(...)`
: A class-level declaration of the HIT options the block accepts —
  one entry per input variable, output variable, scalar option,
  parameter, or sub-object dependency. Lives in {mod}`neml2.schema`.

`from_hit(cls, node, factory)`
: A classmethod that turns a parsed HIT node into an instance.
  Models with a `HitSchema` inherit a default implementation from
  `Model.from_hit` that needs no overriding for the common case;
  models with dynamic I/O or non-trivial construction logic override
  it. We rely on the inherited version on this page.

## The example model

We will scaffold the same projectile model the previous tutorial
declared — a `Vec` velocity in, a `Vec` acceleration out, under
gravity plus linear drag:

$$
\boldsymbol{a} \;=\; \boldsymbol{g} \;-\; \mu\, \boldsymbol{v},
$$

with $\mu$ a calibratable `Scalar` parameter and $\boldsymbol{g}$ a
constant `Vec` buffer. The physics is one line of typed-wrapper
algebra; the rest of the file is the input-file connection that this
tutorial is about.

```{literalinclude} projectile.py
:language: python
:caption: projectile.py
```

Two things to notice:

1. **`@register_native("ProjectileAcceleration")`** is what makes
   `type = ProjectileAcceleration` resolvable from an input file.
   The string passed to the decorator is the HIT type name — it does
   not have to match the Python class name (it usually does, by
   convention).
2. **`hit = HitSchema(...)`** is the entire input-file surface. Each
   field is one of:

   | Field            | Maps to                                                             |
   |------------------|---------------------------------------------------------------------|
   | `input(name, T)` | A variable consumed by `forward`; HIT option of the same name lets the user rename it.  |
   | `output(name, T)`| A variable produced by `forward`; same renaming behavior.           |
   | `parameter(name, T, attr=...)` | A calibratable `nn.Parameter`; the HIT option's value is a literal, a `[Tensors]` cross-reference, or a `[Models]`-output reference.  |
   | `buffer(name, T, attr=..., default=...)` | A non-trainable constant baked into the model (`nn.Buffer`); same HIT spec shapes as `parameter` minus the `[Models]`-output mode. |
   | `option(name, type, attr=..., default=...)` | A plain scalar/string/bool option stored on the instance. |
   | `dependency(name, "get_model", attr=...)`   | A reference to another model by HIT name; resolved by the factory. |

   The `attr=` on `parameter`/`buffer`/`option`/`dependency` is the
   instance attribute the resolved value lands on, available to
   `forward`. For `input`/`output`, the HIT-resolved variable name is
   recorded in the per-instance `input_spec` / `output_spec` dicts.

## The input file

The schema above produces this minimal input block:

```{literalinclude} input.i
:language: ini
:caption: input.i
```

Each line in the `[accel]` block lines up with one schema field:

- `type = ProjectileAcceleration` — the registry lookup key.
- `velocity = 'v'` — the user-facing name to use for the
  `input("velocity", ...)` field. Without this line the canonical
  name `velocity` is used.
- `acceleration = 'a'` — same renaming pattern for the
  `output("acceleration", ...)` field.
- `dynamic_viscosity = 0.05` — a literal scalar consumed by the
  `parameter("dynamic_viscosity", ...)` field. The factory turns it
  into an `nn.Parameter` exposed on the instance as `self.mu` (the
  attribute name we asked for via `attr="mu"`).

## Loading and inspecting

Before `neml2.load_model` can recognise the type name, the module
that defines it must have been imported (so the `@register_native`
decorator has run). In a Python script this is just an `import`; in
this notebook page the source file is sitting next to `input.i`, so
we put its directory on `sys.path` and import it.

```{code-cell} ipython3
import sys
sys.path.insert(0, ".")
import projectile  # the @register_native runs at import time

import neml2
model = neml2.load_model("input.i", "accel")
model
```

The schema-driven name resolution shows up in `input_spec` and
`output_spec`:

```{code-cell} ipython3
model.input_spec, model.output_spec
```

The canonical names declared in the schema (`velocity`, `acceleration`)
were overridden by the HIT values (`v`, `a`). The keys here are what
sibling models inside a `ComposedModel` would use to wire to ours —
covered in [](tutorials-extension-composition).

The parameter declared in the schema is exposed under the `attr`
name we picked, as a typed `Scalar` wrapping an `nn.Parameter`:

```{code-cell} ipython3
model.mu
```

And the forward operator evaluates as expected:

```{code-cell} ipython3
from neml2.types import Vec

model(Vec.fill(10.0, 2.0, 0.0))
```

## Confirming the registry entry

It is sometimes useful to ask the registry directly what type names
it knows about — especially when debugging a "not registered" error
from `load_model`. The registry is a plain dict under
`neml2.factory`:

```{code-cell} ipython3
from neml2.factory import _registry

"ProjectileAcceleration" in _registry, _registry["ProjectileAcceleration"]
```

If a `KeyError` mentioning *"not registered in NativeRegistry"* fires
at `load_model` time, the cause is almost always that the module
holding `@register_native` was never imported — fix the import (or
re-order imports) so the decorator runs before the load call.

## Loading the extension from the CLI

Every `neml2-*` console script accepts a cumulative `--load PATH`
flag for exactly this situation: a custom model defined outside the
`neml2` package can be driven from the shell without writing a
wrapper script. `PATH` is either a filesystem path to a `.py` file
(or a package directory) or a dotted module name on `sys.path`. The
flag is processed before the input file is parsed, so the registered
type is visible by the time `load_model` runs:

```bash
neml2-run --load ./projectile.py input.i driver
neml2-inspect --load ./projectile.py input.i accel
neml2-syntax --load ./projectile.py --type ProjectileAcceleration --json -
neml2-compile --load ./projectile.py input.i --model accel
```

Repeat `--load` to pull in several extensions; they import in the
order given, so a later module may depend on names registered by an
earlier one. See [](cli-utilities) for the option in context.

## What's next

- The schema fields glossed over here — typed inputs, list options,
  parameters that may be promoted to runtime inputs — are covered in
  detail by [](tutorials-extension-arguments).
- The body of `forward` (and its `v_jvp` chain-rule hook for
  composability) is the subject of
  [](tutorials-extension-forward).
- Once your model loads, the next step is usually to compose it with
  other models inside a single `[Models]` block — see
  [](tutorials-extension-composition).
