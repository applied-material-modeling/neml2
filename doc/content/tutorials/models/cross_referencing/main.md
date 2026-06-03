(tutorials-models-cross-referencing)=
# Cross-referencing

Most input files involve more than one object, and those objects need
to refer to each other — a driver names the model it drives, a model
names the tensors it consumes. NEML2's mechanism for this is simple:
**anywhere a field expects an object name, you can write the name of
another section in the file.**

## Referring to a model from a driver

`ModelUnitTest` is a driver that evaluates a model against a fixed
input/output dictionary. Its `model` option doesn't take a model
definition — it takes the *name* of one declared elsewhere:

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
and `input_SR2_values` are *parallel* lists — one entry per input
variable — and every entry in `_values` is a `[Tensors]` reference.

There is no "inline a model" alternative — `model = '<...>'` *must*
be a name. The same is true for `TransientDriver`, `Verification`,
and every other driver type:

```ini
[Drivers]
  [run]
    type  = TransientDriver
    model = 'chaboche_voce_perzyna'                   # ← name of the [Models] entry
    prescribed_time   = 'times'
    prescribed_strain = 'strains'
  []
[]
```

`prescribed_time = 'times'` is itself a cross-reference — `times` is
the name of a `[Tensors]` entry, covered next.

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

But for a simple literal, **NEML2 also accepts the literal in place**
— no `[Tensors]` section needed:

```ini
[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients      = '200e3          0.3'               # inline literals
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
```

So when would you go through `[Tensors]`? Three common reasons:

1. **Sharing.** The same value is consumed by multiple models and you
   want to update it in one place.
2. **Non-literal construction.** The value comes from a torch
   expression (`torch.linspace(...)`), a CSV file, or a computed
   combination of other tensors — none of which can be written as a
   bare literal in the consumer's field.
3. **Sub-batch structure.** `.sub_batch.retag(n)` tags trailing dims as
   the sub-batch axis. A bare literal can't carry that metadata.

A representative non-trivial entry, the temperature-controls axis of a
lookup table:

```ini
[Tensors]
  [T_controls]
    type = Python
    expr = 'Scalar(torch.linspace(300.0, 1200.0, 20, dtype=torch.float64)).sub_batch.retag(1)'
  []
[]
```

Once declared, every model that references `T_controls` shares it.

## Where to go next

The same name-binding mechanism is what `ComposedModel` uses to wire
its children together — see [](tutorials-models-composition).
