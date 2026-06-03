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

(tutorials-models-composition)=
# Model composition

## Why compose?

Almost any non-trivial constitutive theory is a chain of small maps.
A Perzyna-type viscoplastic model, for instance, threads together

$$
\begin{align*}
  \boldsymbol{\varepsilon}^e &= \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^p, \\
  \boldsymbol{\sigma} &= 3K\operatorname{vol}\boldsymbol{\varepsilon}^e + 2G\operatorname{dev}\boldsymbol{\varepsilon}^e, \\
  \bar{\sigma} &= J_2(\boldsymbol{\sigma}), \\
  f &= \bar{\sigma} - \sigma_y, \\
  \boldsymbol{N} &= \partial f/\partial\boldsymbol{\sigma}, \\
  \dot{\gamma} &= \left(\langle f\rangle / \eta\right)^n, \\
  \dot{\boldsymbol{\varepsilon}}^p &= \dot{\gamma}\,\boldsymbol{N}.
\end{align*}
$$

Every one of those constitutive choices — small vs. finite strain,
linear vs. nonlinear elasticity, presence or absence of hardening, the
shape of the rate sensitivity — has multiple variants in the catalog.
If NEML2 shipped a monolithic class for every combination the source
tree would be astronomically large. Instead, each box on the right-hand
side is its own `Model` and `ComposedModel` glues a chosen set
together.

The same decomposition also buys you modularity: every box is a
self-contained `Model` that can be tested, calibrated, and swapped
independently. The usual cost of that modularity — Python-side
dispatch overhead on every step through every sub-model — is removed
once the composed graph is exported through NEML2's compilation
pipeline (see [](tutorials-models-compiled)), so you don't have to
pick between a tidy theory-aligned decomposition and a fast hot loop.

## A worked example

To keep the wiring visible we'll use three small models from the
catalog instead of the full plasticity stack:

$$
\begin{align}
  \bar{a} &= I_1(\boldsymbol{a}), \\
  \bar{b} &= J_2(\boldsymbol{b}), \\
  \dot{\boldsymbol{b}} &= \bar{b}\,\boldsymbol{a} + \bar{a}\,\boldsymbol{b}.
\end{align}
$$

The first two equations are scalar invariants of symmetric tensors —
[](models-SR2Invariant) does both with an `invariant_type` switch.
The third is a linear combination of two `SR2` tensors with scalar
weights — [](models-SR2LinearCombination).

```{literalinclude} input.i
:language: ini
:caption: input.i
```

The trick is in `eq3`'s `weights = 'b_bar a_bar'`. `b_bar` and `a_bar`
are not literals and not `[Tensors]` entries — they are the *output
variable names* of `eq2` and `eq1`. When `ComposedModel` walks its
children it sees that `eq3` consumes two scalars that `eq1` and `eq2`
produce, and wires them up.

## Inspecting the wiring

Before evaluating anything, ask `neml2-inspect` to print the resolved
input/output graph of the composed model:

```{code-cell} ipython3
import subprocess
print(subprocess.run(
    ["neml2-inspect", "input.i", "eq"],
    capture_output=True, text=True, check=True,
).stdout)
```

Three things to notice:

1. **Inputs collapsed to `a` and `b`.** The dependency resolver
   identified the two unbound input variables and surfaced them as the
   composed model's inputs. The intermediate scalars `a_bar` and
   `b_bar` are no longer free inputs — they're produced internally.
2. **Outputs collapsed to `b_rate`.** Same idea, in reverse: `a_bar`
   and `b_bar` are consumed downstream, so they don't appear as
   outputs of the composed model. (If you need them, list them under
   `additional_outputs` on the `ComposedModel`.)
3. **Parameters collapsed to `eq3.offset`.** `eq3.weight_0` and
   `eq3.weight_1` are gone — they've been replaced by the producer
   links from `eq2` and `eq1`. Only the literal `offset = 0` survives
   as a free parameter.

Running `neml2-inspect` whenever you wire a new composed model is the
fastest way to catch typos in variable names. A name mismatch shows up
here as either an extra dangling input ("why is `a_bar` still listed
as an input?") or a missing output, and it's much easier to read than
a shape mismatch deep inside `__call__`.

## Loading and evaluating the composed model

From Python the composed model behaves like any other `Model` — load
it with `neml2.load_model` and call it on its inputs:

```{code-cell} ipython3
import torch
import neml2
from neml2.types import SR2

torch.set_default_dtype(torch.float64)
eq = neml2.load_model("input.i", "eq")
eq
```

The `input_spec` / `output_spec` properties echo what `neml2-inspect`
showed:

```{code-cell} ipython3
list(eq.input_spec.keys()), list(eq.output_spec.keys())
```

To evaluate, pass the inputs as an `{name: SR2}` dict to
`call_by_name`:

```{code-cell} ipython3
a = SR2(torch.tensor([0.1, 0.05, -0.03, 0.02, 0.06, 0.03]))
b = SR2(torch.tensor([100.0, 20.0, 10.0, 5.0, -30.0, -20.0]))
eq.call_by_name({"a": a, "b": b})
```

The composed model evaluated `eq1`, then `eq2`, then `eq3` (the only
order that respects the producer/consumer dependencies), threaded the
intermediate scalars through `eq3`'s weight slots, and returned
`b_rate`.

## The same thing without `ComposedModel`

To see what `ComposedModel` is buying you, here is the same
calculation done by hand against three standalone sub-models. The
input file is the same three `[Models]` entries with `weights = '1 1'`
on `eq3` so that `weight_0` and `weight_1` stay as free parameters:

```{literalinclude} input_manual.i
:language: ini
:caption: input_manual.i
```

```{code-cell} ipython3
import torch.nn as nn

eq1 = neml2.load_model("input_manual.i", "eq1")
eq2 = neml2.load_model("input_manual.i", "eq2")
eq3 = neml2.load_model("input_manual.i", "eq3")

# 1. Evaluate the two invariants.
a_bar = eq1(a)
b_bar = eq2(b)

# 2. Manually wire the weights of eq3 to those intermediate values.
eq3.weight_0 = nn.Parameter(b_bar.data)
eq3.weight_1 = nn.Parameter(a_bar.data)

# 3. Evaluate eq3 to get b_rate.
eq3(a, b)
```

The result agrees with the composed version, but the caller had to:

- decide which model to evaluate first,
- remember which weight slot maps to which invariant, and
- physically copy the intermediate values into `eq3`'s parameters.

Three models is manageable. Three dozen — with shape-checked tensors
and parameter sharing — is not. `ComposedModel` does this bookkeeping
once at load time, then disappears.

:::{note}
The producer/consumer wiring works because `eq3`'s `weights` option
accepts a list of *names* that can resolve to either parameters, the
outputs of sibling models, or `[Tensors]` entries. The general story
about parameters-as-cross-references is the subject of
[](tutorials-models-parameters-revisited).
:::

## Where to go next

- The flexible name-binding used for `eq3`'s `weights` is generalized
  in [](tutorials-models-parameters-revisited).
