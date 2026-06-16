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

(tutorials-models-parameters-revisited)=
# Model parameters revisited

You'll take a small thermo-elastic model and try four ways of
specifying its parameters — a literal number, a shared tensor, another
sub-model, and a runtime input. Each one is a one-line change to the
input file.

## The physics

The toy constitutive law has a thermal eigenstrain feeding linear
elasticity:

$$
  \boldsymbol{\varepsilon}^g = \alpha (T - T_0)\,\boldsymbol{I},
  \qquad
  \boldsymbol{\varepsilon}^e = \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^g,
  \qquad
  \boldsymbol{\sigma}
    = 3K\,\operatorname{vol}\boldsymbol{\varepsilon}^e
    + 2G\,\operatorname{dev}\boldsymbol{\varepsilon}^e,
$$

with the moduli expressed through Young's modulus $E$ and Poisson's
ratio $\nu$. NEML2 ships one model per equation —
`ThermalEigenstrain`, `SR2LinearCombination`, and
`LinearIsotropicElasticity` — and a `ComposedModel` glues them
together by matching producer outputs to consumer inputs.

## Parameters from literals

The baseline input file sets every parameter to a numeric literal:

```{literalinclude} input1.i
:language: ini
:caption: input1.i
```

Load it and inspect what we got:

```{code-cell} ipython3
import neml2

model = neml2.load_model("input1.i", "eq")
model
```

```{code-cell} ipython3
print("inputs:", {n: t.__name__ for n, t in model.input_spec.items()})
print("outputs:", {n: t.__name__ for n, t in model.output_spec.items()})
print("parameters:")
for name, p in model.named_parameters():
    print(f"  {name:20s} shape={list(p.shape)}  value={p.detach().tolist()}")
```

A couple of things to notice:

1. Each child's parameters show up prefixed by its HIT name
   (`eq1.`, `eq2.`, `eq3.`).
2. `weight_0`, `weight_1`, and `offset` came along for the ride.
   `weights = '1 -1'` on `eq2` got stored as two parameters, and
   `offset` defaulted to `0` — anything a child exposes as a
   parameter is a parameter of the composition too, even when you
   never named it.

Gradients flow through these like any other PyTorch module:

```{code-cell} ipython3
import torch
from neml2.types import SR2, Scalar

strain = SR2.fill(0.01, 0.0, 0.0, 0.0, 0.0, 0.0)
T = Scalar(350.0)

(stress,) = model(T, strain)
stress.data.sum().backward()

print("dL/dE     =", model.eq3.E.data.grad.item())
print("dL/dnu    =", model.eq3.nu.data.grad.item())
print("dL/dalpha =", model.eq1.alpha.data.grad.item())
```

A composed model returns one entry per output — here there's just
`stress`, so we destructure the singleton tuple.

## Sharing values via the `[Tensors]` section

A literal is fine for a single number, but if you want a batched
value or you want several models to share the same value, write it
once under `[Tensors]` and refer to it by name:

```{literalinclude} input2.i
:language: ini
:caption: input2.i
```

`alpha` is now a `(2, 2)`-batched scalar. Every entry of `eq1.alpha`
gets its own value, and every downstream tensor picks up the leading
`(2, 2)` batch shape:

```{code-cell} ipython3
import torch
import neml2
from neml2.types import SR2, Scalar

model = neml2.load_model("input2.i", "eq")

print("eq1.alpha:")
print(model.eq1.alpha.data)
print()
strain = SR2.fill(0.01, 0.0, 0.0, 0.0, 0.0, 0.0)
T = Scalar(350.0)
(stress,) = model(T, strain)
print("stress.data.shape:", tuple(stress.data.shape))
```

`eq1.alpha` is still a parameter of the composed model — it just got
its initial value from a batched tensor instead of a literal. See
[](tutorials-models-cross-referencing) for more on what `[Tensors]`
can do.

## Parameters as sub-model outputs

A `[Tensors]` value is still static — it doesn't change as the model
runs. For thermomechanical coupling we want $\alpha(T)$ and $E(T)$ to
be computed from the current temperature on each call. To do that,
point the parameter at *another model in the file*. Here we add a
`ScalarLinearInterpolation` that maps `temperature` to a value, and
reference it by name:

```{literalinclude} input3.i
:language: ini
:caption: input3.i
```

The composition picks up two new children and loses `eq1.alpha` and
`eq3.E` from its parameter list — those slots are now filled by the
interpolants, whose own parameters (the abscissa/ordinate vectors)
take their place:

```{code-cell} ipython3
import neml2

model = neml2.load_model("input3.i", "eq")
model
```

```{code-cell} ipython3
print("inputs:", {n: t.__name__ for n, t in model.input_spec.items()})
print("parameters:")
for name, p in model.named_parameters():
    print(f"  {name:20s} shape={list(p.shape)}")
```

NEML2 worked out from the variable names alone that `alpha` and `E`
needed to be evaluated before `eq1` and `eq3` — there's no extra
wiring to declare.

Calling the model still takes only $(T, \boldsymbol{\varepsilon})$,
because the new sub-models read `temperature` from the same input
slot the eigenstrain model already uses:

```{code-cell} ipython3
import torch
from neml2.types import SR2, Scalar

strain = SR2.fill(0.01, 0.0, 0.0, 0.0, 0.0, 0.0)
T = Scalar(350.0)
(stress,) = model(T, strain)
stress
```

:::{note}
If the referenced model has more than one output, the variable
specifier is ambiguous. Use the dotted form
`<model_name>.<variable_name>` (e.g. `'mymodel.eigenstrain'`) to
pick a specific output.
:::

## Parameters as runtime inputs

One last twist: if you put a bare name in the parameter slot that
doesn't match any tensor or model in the file, NEML2 promotes that
parameter to an input — you'll supply it at call time instead.

```{literalinclude} input4.i
:language: ini
:caption: input4.i
```

The composition now has three inputs instead of two:

```{code-cell} ipython3
import neml2

model = neml2.load_model("input4.i", "eq")
print("inputs:", list(model.input_spec))
print("parameters:")
for name, p in model.named_parameters():
    print(f"  {name:20s} shape={list(p.shape)}")
```

`eq1.alpha` is gone from the parameter list, and `alpha` shows up as
an input slot instead. The forward call now takes a third argument,
in the order shown by `input_spec`:

```{code-cell} ipython3
import torch
from neml2.types import SR2, Scalar

strain = SR2.fill(0.01, 0.0, 0.0, 0.0, 0.0, 0.0)
T = Scalar(350.0)
alpha = Scalar(1.2e-5)
(stress,) = model(T, alpha, strain)
stress
```

All four cases above go through the same syntactic slot — what
NEML2 does depends only on what the name resolves to:

| The name `x` in `CTE = 'x'`    | Result on the composition                                    |
| ------------------------------ | ------------------------------------------------------------ |
| literal number (e.g. `'1e-6'`) | `eq1.alpha` is a parameter initialized from the literal      |
| `[Tensors/x]` entry            | `eq1.alpha` is a parameter initialized from the tensor       |
| `[Models/x]` entry             | `x` becomes a child; `eq1.alpha` is replaced by `x`'s output |
| no other match                 | `x` is added as an input variable of the composition         |

The consuming model doesn't know or care which row applied.

## Where to go next

- [`neml2-inspect`](cli-neml2-inspect) prints the same input and
  parameter info straight from an input file — handy for checking
  the wiring without writing any Python.
- [](tutorials-models-composition) walks through how NEML2 decides
  what to evaluate when.
- [](tutorials-models-implicit-model) reuses the same parameter
  machinery to expose state-dependent Jacobians to the Newton
  solver.
