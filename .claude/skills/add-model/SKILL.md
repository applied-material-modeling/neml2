---
name: add-model
description: Scaffold a new NEML2 `Model` subclass under `neml2/models/<domain>/`. NEML2 models are pure Python classes deriving from `neml2.model.Model` (an `nn.Module`), declared via a `HitSchema`, decorated with `@register_native("TypeName")`, and tested with a `.i` driver under `tests/models/<domain>/`. Trigger on any "add a Foo model", "scaffold a new yield function", "create an SR2-to-SR2 mapping", etc.
---

# add-model

A NEML2 model is a Python class with three load-bearing pieces:

1. `@register_native("TypeName")` decorator (from `neml2.factory`).
2. `hit = HitSchema(input(...), output(...), parameter(...))` block from
   `neml2.schema`.
3. `forward(self, *typed_inputs, v=None, v2=None, vh=None)` body in
   typed-wrapper algebra.

## File placement

- Source: `neml2/models/<domain>/<TypeName>.py`. Import it from the
  domain's `__init__.py` so the decorator fires on package load.
- Test driver: `tests/models/<domain>/<TypeName>.i` — a HIT
  `ModelUnitTest` fixture discovered automatically by
  `tests/test_model_unit_tests.py`.

## Skeleton

```python
# neml2/models/solid_mechanics/plasticity/MyHardening.py
from neml2.factory import register_native
from neml2.model import Model
from neml2.schema import HitSchema, input, output, parameter
from neml2.chain_rule import ChainRuleDict
from neml2.types import Scalar


@register_native("MyHardening")
class MyHardening(Model):
    r"""h = K * ep — single-line statement of the forward."""

    hit = HitSchema(
        input("equivalent_plastic_strain", Scalar),
        output("isotropic_hardening", Scalar),
        parameter("hardening_modulus", Scalar, attr="K", allow_nonlinear=True),
    )

    K: Scalar  # populated by from_hit auto-declaration

    def forward(  # type: ignore[override]
        self,
        ep: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        K = self._get_param("K", nl_params, Scalar)
        h = K * ep
        if v is None:
            return h
        actions_1 = {"equivalent_plastic_strain": lambda V: K * V}
        return h, self.apply_chain_rule(v, "isotropic_hardening", actions_1, output=h)
```

## Canonical patterns

| Shape | Reference leaf |
| :--- | :--- |
| Linear / reduction | `LinearIsotropicElasticity`, `SumSlipRates` |
| Rotation / kinematic product rule | `ElasticStrainRate`, `RotationMatrix` |
| Sub-batch coupling (needs `list_deriv = "dense"`) | `IntermediateMean`, FV leaves |
| Time integration (derived I/O names) | `BackwardEulerTimeIntegration` |
| Second-order chain rule (linear) | `LinearCombination`, `YieldFunction` |
| Second-order chain rule (nonlinear) | `SR2Invariant` |
| List-of-parameters from one HIT option | `LinearCombination` (weights) |
| Conditional / optional inputs | `YieldFunction` (`default=None`) |

## Hard rules

- **No raw `torch.<op>(wrapper.data)` in `forward` or chain-rule
  actions.** If the primitive isn't in `neml2.types.functions`, add it
  there as a typed free function first.
- **Don't form the Jacobian; push the tangent.** The action receives a
  tangent of the input type and returns a tangent of the output type,
  in typed-wrapper algebra. The framework accumulates over seed leaves.
- **Docstrings must be ASCII** (the `test_syntax_cli` ASCII guard
  rejects em-dashes, Greek letters, math symbols). Save those for
  comments.
- **No `__init__`** for most leaves — `from_hit` auto-declares every
  `parameter(...)` field. Write one only when the schema can't express
  the construction logic (dynamic I/O, computed defaults, etc.).
- **Canonical variable names match HIT defaults.** A user writes
  `strain = 'elastic_strain'` in HIT and the schema resolves it. The
  `forward` body keeps using canonical names; `apply_chain_rule`
  translates.

## Test fixture

Drop a `<TypeName>.i` under `tests/models/<domain>/` with a
`ModelUnitTest` driver. The test runner picks it up automatically;
no Python wrapper needed.

```ini
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names  = 'equivalent_plastic_strain'
    input_Scalar_values = 'ep'
    output_Scalar_names = 'isotropic_hardening'
    output_Scalar_values = 'h_expected'
  []
[]

[Tensors]
  [ep]
    type = Python
    expr = 'Scalar(0.01)'
  []
  [h_expected]
    type = Python
    expr = 'Scalar(20.0)'
  []
[]

[Models]
  [model]
    type = MyHardening
    hardening_modulus = 2000.0
  []
[]
```

The driver checks forward values + auto-checks first-order JVPs against
`torch.autograd.functional.jvp` of the same forward.

## References

- `neml2/model.py` — base class, `apply_chain_rule`,
  `apply_chain_rule_2`, `propagate_tangents`, `_get_param`.
- `neml2/types/functions.py` — typed free functions and JVP primitives.
- `neml2/schema.py` — `HitSchema`, `input`, `output`, `parameter`,
  `derived_input`, `var_inputs`, `option`.
