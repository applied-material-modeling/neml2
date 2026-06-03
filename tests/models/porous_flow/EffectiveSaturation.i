# Translated from tests/unit/models/porous_flow/EffectiveSaturation.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'phi phimax'
    input_Scalar_values = '0.553456 0.374'
    output_Scalar_names = 'out'
    output_Scalar_values = '1.479828877'
  []
[]

[Models]
  [model]
    type = EffectiveSaturation
    residual_saturation = 0.0
    fluid_fraction = 'phi'
    max_fraction = 'phimax'
    effective_saturation = 'out'
  []
[]
