[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/phi state/phimax'
    input_Scalar_values = '0.553456 0.374'
    output_Scalar_names = 'state/out'
    output_Scalar_values = '1.479828877'
  []
[]

[Models]
  [saturation]
    type = EffectiveSaturation
    residual_saturation = 0.0
    fluid_fraction = 'state/phi'
    max_fraction = 'state/phimax'
    effective_saturation = 'state/out'
  []
  [model]
    type = ComposedModel
    models = 'saturation'
  []
[]
