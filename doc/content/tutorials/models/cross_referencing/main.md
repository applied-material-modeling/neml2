(tutorials-models-cross-referencing)=
# Cross-referencing

You'll wire a driver to a model and a model to its tensor inputs by
name — the basic glue that lets an input file hold more than one
object. Anywhere a field expects an object name, you can write the
name of another section in the file.

## Referring to a model from a driver

`ModelUnitTest` is a driver that evaluates a model against a fixed
input. Its `model` option takes the *name* of a model declared
elsewhere in the file:

```ini
[Drivers]
  [unit]
    type  = ModelUnitTest
    model = 'elasticity'                  # ← name of the [Models] entry below
    input_SR2_names  = 'strain'
    input_SR2_values = 'strain_value'     # ← name of the [Tensors] entry below
    output_SR2_names = 'stress'
  []
[]

[Tensors]
  [strain_value]
    type = Python
    expr = 'SR2.fill(0.01, 0.0, 0.0, 0.0, 0.0, 0.0)'
  []
[]

[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients      = '200e3          0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
```

Two cross-references in one block: `model = 'elasticity'` names the
`[Models]` entry to evaluate, and `input_SR2_values = 'strain_value'`
names the `[Tensors]` entry that supplies the input. `input_SR2_names`
and `input_SR2_values` are parallel lists — one entry per input — and
every entry in `_values` is either a `[Tensors]` reference or (for
`Scalar` inputs) an inline number.

Other drivers work the same way:

```ini
[Drivers]
  [run]
    type  = TransientDriver
    model = 'chaboche_voce_perzyna'                   # ← name of the [Models] entry
    prescribed_time  = 'times'                        # ← name of a [Tensors] entry
    force_SR2_names  = 'E'
    force_SR2_values = 'strains'                      # ← name of a [Tensors] entry
  []
[]
```

`prescribed_time = 'times'` is itself a cross-reference — `times` is
the name of a `[Tensors]` entry. Driving forces are supplied as
parallel `force_<Type>_names` / `force_<Type>_values` lists, where
each value token is again a `[Tensors]` name (or an inline literal for
`Scalar`).

## Referring to a tensor from a model

When a model field expects a tensor value, you can also point it at a
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

So when would you go through `[Tensors]`? When the literal won't do —
typically because you want to share the value across several models,
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

## Where to go next

The same name-binding mechanism is what `ComposedModel` uses to wire
its children together — see [](tutorials-models-composition).
