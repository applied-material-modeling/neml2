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

(tutorials-extension-composition)=
# Composing your model with others

The previous three tutorials walked through writing a fresh `Model`:
[](tutorials-extension-arguments) declared its inputs, outputs, and
parameters; [](tutorials-extension-input-files) registered it so the
HIT parser can see it; [](tutorials-extension-forward) implemented
the math. The payoff for all that scaffolding is that your model now
behaves like every other entry in the catalog — meaning it can be
dropped into a [`ComposedModel`](models-ComposedModel) alongside built-ins
and have NEML2 resolve the wiring automatically.

This page shows that composition end-to-end. The pattern is the same
as the one introduced in the broader [](tutorials-models-composition)
tutorial; the focus here is specifically on **your own custom model
joining the composition** and on the two knobs that matter most when
you do — `additional_outputs` for surfacing intermediates, and
`neml2-inspect` for verifying the wiring before you call anything.

## The setup

A composed model is just an aggregate of sibling `[Models]` entries
that share variable names. Imagine your fresh model is a stress
predictor of some kind — say the
[`LinearIsotropicElasticity`](models-LinearIsotropicElasticity) we built up
in the previous tutorials — and you want to chain it with the
catalog's [`SR2Invariant`](models-SR2Invariant) to get the von Mises stress
as a single composed forward operator:

$$
\begin{align}
  \boldsymbol{\sigma} &= 3K\,\operatorname{vol}\boldsymbol{\varepsilon}^e
                       + 2G\,\operatorname{dev}\boldsymbol{\varepsilon}^e, \\
  \bar{\sigma}        &= \sqrt{\tfrac{3}{2}\,
                          \operatorname{dev}\boldsymbol{\sigma}
                          :\operatorname{dev}\boldsymbol{\sigma}}.
\end{align}
$$

:::{note}
We use `LinearIsotropicElasticity` as a stand-in for "your custom
model from the previous tutorials" so this page is runnable as-is.
Substitute whatever class you wrote — the wiring story is identical
for any registered `Model`.
:::

## The input file

```{literalinclude} input.i
:language: ini
:caption: input.i
```

Three things to notice in the `[Models]` block:

1. **Both models live side by side** under `[Models]`, exactly as they
   would in any normal input file. Your custom model is no different
   from `SR2Invariant` here — the registry doesn't distinguish.
2. **A third entry of `type = ComposedModel`** lists the two children
   in its `models` field and is what gets named (`chain`) and loaded
   from Python. The composed model is itself just another `Model`.
3. **The wiring is implicit, through variable names.** `elasticity`
   produces a variable called `stress`; `vonmises` consumes a variable
   called `tensor` that we renamed to `stress`. The dependency
   resolver sees the producer/consumer match and threads the value
   through internally — `stress` is no longer a free input of the
   composed model.

## Exposing an intermediate output

By default, an intermediate variable that flows between children of a
`ComposedModel` is *hidden* — only the leaf outputs survive on the
composed model's output spec. That's the right default (it keeps the
external surface minimal), but sometimes you want a downstream
consumer or postprocessor to also see an intermediate. The
`additional_outputs` option on `ComposedModel` does exactly that:

```ini
[chain]
  type = ComposedModel
  models = 'elasticity vonmises'
  additional_outputs = 'stress'   # surface the intermediate
[]
```

With `stress` listed under `additional_outputs`, it stays an internal
producer/consumer link **and** appears on the composed model's output
spec, so a single forward call returns both `vm_stress` and `stress`.

## Inspecting the wiring

Before evaluating anything, ask `neml2-inspect` to print the
resolved input/output graph. This is the same diagnostic step used in
[](tutorials-models-composition), and it's worth running every time
you add or rename a variable — name mismatches surface here as obvious
dangling inputs instead of cryptic shape errors deep in the forward
operator:

```{code-cell} ipython3
!neml2-inspect input.i chain
```

Read the output top to bottom:

- **Inputs (1).** `elastic_strain` is the only unbound input — `stress`
  is no longer free because `elasticity` produces it internally.
- **Outputs (2).** `vm_stress` is the leaf output of `vonmises`;
  `stress` is the intermediate we explicitly surfaced via
  `additional_outputs`. Drop `additional_outputs` and you'd see only
  `vm_stress` here.
- **Parameters (2).** `elasticity.E` and `elasticity.nu` are namespaced
  under the child name — that's how you reach them from Python once
  the composed model is loaded.

A wiring bug — say a typo in `tensor = 'stres'` on `vonmises` —
would manifest in this output as an extra unbound `stres` input and a
missing `vm_stress` output (because the resolver couldn't satisfy
`vonmises`'s `tensor` input). Catching it here takes a second; catching
it from a `__call__` traceback can take much longer.

## Loading and evaluating

From Python the composed model behaves like any other `Model` — load
it with `neml2.load_model` and call it:

```{code-cell} ipython3
import torch
import neml2
from neml2.types import SR2

torch.set_default_dtype(torch.float64)
chain = neml2.load_model("input.i", "chain")
chain
```

The `repr` shows the child models as registered submodules — the
composed model is a `torch.nn.Module` whose children are the wired-up
sibling models. The input and output specs match what `neml2-inspect`
reported:

```{code-cell} ipython3
list(chain.input_spec.keys()), list(chain.output_spec.keys())
```

To evaluate, pass the inputs as a `{name: SR2}` dict to
`call_by_name`. Uniaxial elastic strain along the x-axis, with
$\nu = 0.3$, gives a stress state with $\sigma_{xx}=1$ and
$\sigma_{yy}=\sigma_{zz}\approx 0$, hence $\bar\sigma = 1$:

```{code-cell} ipython3
strain = SR2.fill(0.01, -0.003, -0.003, 0.0, 0.0, 0.0)
chain.call_by_name({"elastic_strain": strain})
```

Both outputs come back from the same call: the leaf `vm_stress` *and*
the intermediate `stress` we surfaced through `additional_outputs`.

## Where to go next

- [](tutorials-models-composition) walks through composition more
  broadly — multi-step chains, parameter binding via output names,
  and the producer/consumer rules the dependency resolver follows.
  This page covers the same machinery from the custom-model angle;
  that page covers the wider story.
- Composed models are themselves `Model`s, so they can be
  exported, compiled to AOT-Inductor, or recursively re-composed
  inside larger graphs without any changes on the consumer side.
