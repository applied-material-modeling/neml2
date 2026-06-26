(tutorials-models-cross-referencing)=
# Cross-referencing

In this tutorial, we'll wire a model to its tensor inputs and to other
models by name — the basic glue that lets an input file hold more than
one object. Anywhere a field expects an object name, we can write the
name of another section in the file.

## Referring to a tensor from a model

When a model field expects a tensor value, we can point it at a
`[Tensors]` entry by name:

```ini
[Tensors]
  [E]
    type = Python
    expr = 'Scalar(200e3)'
  []
  [nu]
    type = Python
    expr = 'Scalar(0.3)'
  []
[]

[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients      = 'E              nu'                # ← refers to [Tensors] entries
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
```

But for simple scalar literals, **many model fields accept the
number directly** — no `[Tensors]` section needed. (Non-scalar
tensor inputs still need a `[Tensors]` entry.)

```ini
[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients      = '200e3          0.3'               # inline literals
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
```

So when would we go through `[Tensors]`? When the literal won't do —
typically because we want to share the value across several models,
or because it comes from a torch expression like `torch.linspace(...)`
or a CSV file rather than a bare number.

Here's a temperature-controls axis built from a torch expression:

```ini
[Tensors]
  [T_controls]
    type = Python
    expr = 'Scalar.linspace(300.0, 1200.0, 20).sub_batch.retag(1)'
  []
[]
```

Once declared, every model that references `T_controls` shares it.

## Referring to a model from another model

Some models operate on *another model* rather than on a tensor, and
their model-valued field takes the name of a `[Models]` entry the same
way. `Normality` is one: it differentiates a scalar-valued function
produced by another model. Here it wraps the von Mises stress invariant
to produce the associated flow direction
$\boldsymbol{N} = \partial \sigma_\mathrm{eff} / \partial \boldsymbol{\sigma}$:

```ini
[Models]
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor    = 'mandel_stress'
    invariant = 'effective_stress'
  []
  [normality]
    type = Normality
    model    = 'vonmises'          # ← name of the [Models] entry above
    function = 'effective_stress'  # the scalar output of `vonmises` to differentiate
    from     = 'mandel_stress'
    to       = 'flow_direction'
  []
[]
```

`model = 'vonmises'` is the cross-reference: `normality` doesn't redefine
the invariant, it points at the existing `[Models]` entry by name and
differentiates its `effective_stress` output. Note that `function`,
`from`, and `to` are *variable* names, not section names — they pick out
inputs and outputs by the variables a model produces and consumes, which
is the wiring mechanism the next tutorial builds on.

## Where to go next

The same name-binding mechanism is what `ComposedModel` uses to wire
its children together — see [](tutorials-models-composition).
