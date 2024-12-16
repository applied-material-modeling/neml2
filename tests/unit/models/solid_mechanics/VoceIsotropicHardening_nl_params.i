[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/ep params/R params/d'
    input_Scalar_values = '0.1 100.0 1.1'
    output_Scalar_names = 'state/internal/k'
    output_Scalar_values = '10.416586470347177'
  []
[]

[Models]
  [R]
    type = ScalarInputParameter
    from = 'params/R'
  []
  [d]
    type = ScalarInputParameter
    from = 'params/d'
  []
  [model0]
    type = VoceIsotropicHardening
    saturated_hardening = 'R'
    saturation_rate = 'd'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
