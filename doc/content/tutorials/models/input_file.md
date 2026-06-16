(tutorials-models-input-file)=
# Input files

You'll learn enough of NEML2's input-file syntax to read and write the
files you'll see throughout the rest of the tutorials. The same file
is parsed by Python (`neml2.load_input`) and by the CLI tools
(`neml2-run`, `neml2-inspect`, …).

NEML2 uses the [HIT](https://github.com/applied-material-modeling/neml2-hit)
format. This tutorial covers the syntax you'll use day-to-day; the
[HIT README](https://github.com/applied-material-modeling/neml2-hit/blob/main/README.md)
is the authoritative reference for the rest.

## Anatomy

A HIT file is a sequence of **sections** and **fields**. A section
opens with `[name]` and closes with `[]`; a field is a `key = value`
pair.

```ini
# Lines starting with `#` are comments.
[section]
  foo = 1                   # an integer
  bar = 3.14159             # a float
  baz = 'string value'      # a string
  qux = 'a b c'             # a 1-D array (whitespace-separated, quoted)
  [nested_section]
    type = SomeRegisteredType
  []
[]
```

Indentation is purely visual — HIT doesn't care.

## Sections and systems

A NEML2 input file is organized into a few **top-level sections** —
`[Models]`, `[Tensors]`, `[Solvers]`, `[Drivers]`, and a few others
(see the [syntax catalog](syntax-catalog) for the full list).

Inside each top-level section, every nested section describes one
object. Its `type = ...` field names the registered class. For example:

```ini
[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients      = '200e3          0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
```

defines an object named `elasticity` in the `[Models]` section, whose
`type` is `LinearIsotropicElasticity`. The remaining fields are the
options that type declares — see
[](models-LinearIsotropicElasticity) for what each one means.

:::{note}
Section order does not matter. Forward references (consuming a name
defined later in the file) work; NEML2 resolves everything after the
full file is parsed.
:::

## Value types

HIT itself stores every value as text; the *type* is enforced at the
point where NEML2 reads the option. The supported primitive types are:

| Kind      | Examples                                          |
| :-------- | :------------------------------------------------ |
| Integer   | `n = 42`, `n = -7`                                |
| Float     | `x = 3.14`, `x = -1.0e-3`                         |
| Boolean   | `flag = true`, `flag = false` (lowercase only)    |
| String    | `name = 'elasticity'` (single line)               |
| 1-D array | `values = '1 2 3'`                                |
| 2-D array | `matrix = '1 2 3; 4 5 6'` (rows separated by `;`) |

On top of these, NEML2 adds a couple of custom value types worth
calling out:

- **Tensor shape** — parenthesized, comma-separated, no spaces.
  `shape = (5,6,7)` parses to the tuple `(5, 6, 7)`. The empty shape
  `()` is also valid.
- **Device** — `cpu`, `cuda`, `cuda:0`, etc., following
  `<type>[:<index>]`.

## Comments

`#` starts a comment that runs to end of line. It can appear on a
line of its own, after a section header (`[Models] # ok`), or after a
**quoted** value (`coefficients = '200e3 0.3' # ok`). After an
**unquoted** value the `#` is treated as part of the value, so
`foo = 1 # bad` parses the value as `1 # bad`. Quoting the value is
the safe default in NEML2 anyway.

## Includes

A file can splice in another file at any point:

```ini
!include shared/material.i
```

The path is resolved relative to the file containing the directive,
and the included content is parsed as if it had been inlined. Useful
for sharing a `[Tensors]` section across several driver inputs.

## Variable substitution

Values can interpolate other fields or environment variables using
`${...}`:

```ini
E = 200e3
nu = 0.3

[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '${E} ${nu}'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]

# Pull from the environment / chain pieces together:
prefix = /opt/neml2
lib = ${raw ${prefix} /lib}  # → /opt/neml2/lib
home = ${env HOME}           # value of the HOME env var, "" if unset
```

The substitution happens at value-extraction time, so the right-hand
side can sit anywhere in the file or in an included file.

## Where to go next

- The [HIT README](https://github.com/applied-material-modeling/neml2-hit/blob/main/README.md)
  for the corners not covered here: override semantics (`:=`),
  verbatim triple-quoted strings, 2-D array nuances, the full grammar.
- The [syntax catalog](syntax-catalog) for the option list of every
  registered type — the canonical answer to "what fields go in this
  section?"
- The next tutorial, [](tutorials-models-running-your-first-model),
  walks through loading and evaluating the model defined in an input
  file.
