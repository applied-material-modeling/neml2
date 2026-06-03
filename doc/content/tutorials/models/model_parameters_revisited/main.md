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

[](tutorials-models-parameters) showed how to read and mutate a
parameter on a single, standalone model. Once you start
[composing](tutorials-models-composition) models — and almost every
real material model is a composition — the parameter surface gets a
few extra moving parts:

- Each child model contributes its own parameters, and the composed
  model namespaces them under the child's HIT name.
- The same parameter can be set from a `[Tensors]` entry, so several
  consumers can share one value and one update site.
- The same syntax that wires sub-models together can also turn a
  parameter into a *sub-model output* (e.g. a temperature-dependent
  modulus) or a *runtime input variable*, which restructures what the
  composition's input signature even is.

This page works through each of those by varying one input file.

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

Two things worth pointing out:

1. **The composed parameter set is the union of the child parameter
   sets**, and every parameter name is prefixed by the child's HIT
   object name (`eq1.`, `eq2.`, `eq3.`).
2. **`weight_0`, `weight_1`, and `offset` came along for the ride.**
   The user set `weights = '1 -1'` on `eq2`, which NEML2 stored as two
   parameters; `offset` defaulted to `0`. Anything a child model
   exposes as a parameter is parameter material on the composition,
   too — even when you never named it.

Because parameters are stored as `torch.nn.Parameter` instances under
the hood, gradients flow through them on the backward pass exactly
like any other module's weights:

```{code-cell} ipython3
import torch
from neml2.types import SR2, Scalar

strain = SR2(torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))
T = Scalar(350.0)

(stress,) = model(T, strain)
stress.data.sum().backward()

print("dL/dE     =", model.eq3.E.data.grad.item())
print("dL/dnu    =", model.eq3.nu.data.grad.item())
print("dL/dalpha =", model.eq1.alpha.data.grad.item())
```

`ComposedModel.forward` returns a tuple (one entry per output
variable) — here we destructure to a single `stress`.

## Sharing values via the `[Tensors]` section

A numeric literal is the simplest possible parameter spec but it can
only encode one number per scalar slot. If you want a batched value,
or you want several models to share the same value, write the value
once under `[Tensors]` and refer to it by name:

```{literalinclude} input2.i
:language: ini
:caption: input2.i
```

`alpha` is now a `(2, 2)`-batched scalar. The model accepts it without
ceremony — every per-batch entry of `eq1.alpha` gets its own value,
and every downstream tensor inherits the leading `(2, 2)` batch shape:

```{code-cell} ipython3
import torch
import neml2
from neml2.types import SR2, Scalar

model = neml2.load_model("input2.i", "eq")

print("eq1.alpha:")
print(model.eq1.alpha.data)
print()
strain = SR2(torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))
T = Scalar(350.0)
(stress,) = model(T, strain)
print("stress.data.shape:", tuple(stress.data.shape))
```

Conceptually nothing has changed: `eq1.alpha` is still a parameter of
the composed model, it just happens to have been initialized from a
batched tensor instead of a single literal.
[](tutorials-models-cross-referencing) covers the full set of reasons
to reach for `[Tensors]` — sharing between consumers, sub-batch
tagging, and expressions that aren't a bare literal.

## Variable specifiers: parameters as sub-model outputs

Tensor cross-references let parameters be *more interesting values*,
but the values are still static — they don't change as the model
evaluates. For thermomechanical coupling we want $\alpha(T)$ and
$E(T)$ to be computed from the current temperature on every call.

NEML2 expresses that by letting a parameter point at *another model
in the file*. The composed model will then run that sub-model first
and wire its output into the consumer slot. To do so, add a
`ScalarLinearInterpolation` model that maps `temperature` to a value,
and reference it by name:

```{literalinclude} input3.i
:language: ini
:caption: input3.i
```

The composition picks up two new children and loses the original
`eq1.alpha` / `eq3.E` parameters — those slots are now sourced from
the interpolants, whose own parameters (the abscissa/ordinate vectors)
take over:

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

Mathematically, the model went from

$$\boldsymbol{\sigma} = f(\boldsymbol{\varepsilon}, T;\, \alpha, E, \nu)$$

to

$$\boldsymbol{\sigma} = f(\boldsymbol{\varepsilon}, T;\, \mathcal{P}_\alpha, \mathcal{P}_E, \nu),$$

where $\mathcal{P}_\alpha$ and $\mathcal{P}_E$ are whatever
parametrization the interpolant exposes — here the abscissa/ordinate
of a piecewise-linear curve. The dependency resolver figured this out
from the variable names alone; we did not have to declare anywhere
that `alpha` should be evaluated before `eq1`.

Calling the model still takes only $(T, \boldsymbol{\varepsilon})$,
because the new sub-models consume `temperature` from the same input
slot the eigenstrain model already uses:

```{code-cell} ipython3
import torch
from neml2.types import SR2, Scalar

strain = SR2(torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))
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

## Variable specifiers: parameters as runtime inputs

The final form takes the same syntactic slot one step further: write a
bare name that does *not* match any tensor or model in the file, and
NEML2 will assume you want that parameter promoted to an *input
variable* on the composition. The caller then supplies it at
evaluation time.

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

`eq1.alpha` is gone from the parameter list — there is no learnable
state for it any more — and `alpha` shows up as an input slot. The
forward call now takes a third argument, in the order shown by
`input_spec`:

```{code-cell} ipython3
import torch
from neml2.types import SR2, Scalar

strain = SR2(torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))
T = Scalar(350.0)
alpha = Scalar(1.2e-5)
(stress,) = model(T, alpha, strain)
stress
```

This is the same machinery the previous section used to plug in a
sub-model — both modes go through the host model's `allow_nonlinear`
parameter slot. The difference is whether the name matches another
object in the file:

| The name `x` in `CTE = 'x'`    | Result on the composition                                    |
| ------------------------------ | ------------------------------------------------------------ |
| literal number (e.g. `'1e-6'`) | `eq1.alpha` is a parameter initialized from the literal      |
| `[Tensors/x]` entry            | `eq1.alpha` is a parameter initialized from the tensor       |
| `[Models/x]` entry             | `x` becomes a child; `eq1.alpha` is replaced by `x`'s output |
| no other match                 | `x` is added as an input variable of the composition         |

The choice is made once, per parameter, in the input file — nothing
about the consuming model class changes.

## Where to go next

- Reach for [`neml2-inspect`](cli-neml2-inspect) when you want to see
  this resolution play out without writing any Python — it prints
  the same input / parameter axis info, including the wiring
  decisions, straight from the input file.
- The mechanics of how the dependency resolver decides what to
  evaluate when is the subject of
  [](tutorials-models-composition).
- Implicit models ([](tutorials-models-implicit-model)) reuse the
  same parameter machinery to expose state-dependent Jacobians to the
  Newton solver.
