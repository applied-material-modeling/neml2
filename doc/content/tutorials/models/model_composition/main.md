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

You'll wire three small models together into a composed model, inspect
how NEML2 resolves the connections, and evaluate the result. Then we'll
compare it to doing the same plumbing by hand.

Most constitutive theories are a chain of small maps. A Perzyna-type
viscoplastic model, for instance, threads together

\begin{align*}
  \boldsymbol{\varepsilon}^e &= \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^p, \\
  \boldsymbol{\sigma} &= 3K\operatorname{vol}\boldsymbol{\varepsilon}^e + 2G\operatorname{dev}\boldsymbol{\varepsilon}^e, \\
  \bar{\sigma} &= J_2(\boldsymbol{\sigma}), \\
  f &= \bar{\sigma} - \sigma_y, \\
  \boldsymbol{N} &= \partial f/\partial\boldsymbol{\sigma}, \\
  \dot{\gamma} &= \left(\langle f\rangle / \eta\right)^n, \\
  \dot{\boldsymbol{\varepsilon}}^p &= \dot{\gamma}\,\boldsymbol{N}.
\end{align*}

Each line on the right-hand side is its own `Model` in NEML2, and
`ComposedModel` glues a chosen set together. That keeps each piece
testable and swappable, and the Python-side overhead of stepping
through many sub-models disappears once you export the composed graph
through the compilation pipeline (see [](tutorials-models-compiled)).

## A worked example

To keep the wiring visible we'll use three small models from the
catalog instead of the full plasticity stack:

\begin{align}
  \bar{a} &= I_1(\boldsymbol{a}), \\
  \bar{b} &= J_2(\boldsymbol{b}), \\
  \dot{\boldsymbol{b}} &= \bar{b}\,\boldsymbol{a} + \bar{a}\,\boldsymbol{b}.
\end{align}

The first two equations are scalar invariants of symmetric tensors,
handled by [](models-SR2Invariant). The third is a linear combination
of two `SR2` tensors with scalar weights — see
[](models-SR2LinearCombination).

```{literalinclude} input.i
:language: ini
:caption: input.i
```

The trick is in `eq3`'s `weights = 'b_bar a_bar'`. `b_bar` and `a_bar`
aren't literals or `[Tensors]` entries — they're the output names of
`eq2` and `eq1`. `ComposedModel` notices that `eq3` consumes two
scalars that `eq1` and `eq2` produce, and wires them up.

## Inspecting the wiring

Before evaluating anything, ask `neml2-inspect` to print the resolved
graph:

```{code-cell} ipython3
import subprocess
print(subprocess.run(
    ["neml2-inspect", "input.i", "eq"],
    capture_output=True, text=True, check=True,
).stdout)
```

Three things to notice:

1. **Inputs collapsed to `a` and `b`.** The intermediate scalars
   `a_bar` and `b_bar` aren't free inputs — they're produced
   internally.
2. **Outputs collapsed to `b_rate`.** `a_bar` and `b_bar` are consumed
   downstream, so they don't surface as outputs. (If you want them,
   add them under `additional_outputs` on the `ComposedModel`.)
3. **Parameters collapsed to `eq3.offset`.** `eq3.weight_0` and
   `eq3.weight_1` are gone — replaced by the producer links from `eq2`
   and `eq1`. Only the literal `offset = 0` is still free.

Running `neml2-inspect` after wiring a composed model is the fastest
way to catch typos in variable names — a mismatch shows up as a
dangling input or missing output, much easier to read than a shape
error deep inside `__call__`.

## Loading and evaluating the composed model

The composed model loads and calls just like any other model:

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

Under the hood it ran `eq1`, then `eq2`, then `eq3` (the only order
that respects the dependencies), threaded the intermediate scalars
into `eq3`'s weight slots, and returned `b_rate`.

## The same thing without `ComposedModel`

To see what `ComposedModel` is doing for you, here's the same
calculation done by hand. The input file is the same three `[Models]`
entries, but with `weights = '1 1'` on `eq3` so `weight_0` and
`weight_1` stay as free parameters:

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

Same answer, but you had to:

- pick the right evaluation order,
- remember which weight slot maps to which invariant, and
- copy the intermediate values into `eq3`'s parameters by hand.

Three models is manageable. Three dozen isn't. `ComposedModel` does
this bookkeeping once at load time, then gets out of the way.

:::{note}
This works because `eq3`'s `weights` accepts a list of names that can
resolve to a parameter, a sibling model's output, or a `[Tensors]`
entry. See [](tutorials-models-parameters-revisited) for the full
story.
:::

## Where to go next

- The flexible name-binding used for `eq3`'s `weights` is generalized
  in [](tutorials-models-parameters-revisited).
