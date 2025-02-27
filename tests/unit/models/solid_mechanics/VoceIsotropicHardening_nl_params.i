[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/ep state/R state/d'
    input_Scalar_values = '0.1 100.0 1.1'
    output_Scalar_names = 'state/internal/k'
    output_Scalar_values = '10.416586470347177'
  []
[]

[Models]
  [model0]
    type = VoceIsotropicHardening
    saturated_hardening = 'state/R'
    saturation_rate = 'state/d'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
