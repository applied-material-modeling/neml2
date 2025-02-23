[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/gamma_rate state/R state/theta0 state/internal/k'
    input_Scalar_values = '0.1 100.0 110 50.0'
    output_Scalar_names = 'state/internal/k_rate'
    output_Scalar_values = '5.5'
  []
[]

[Models]
  [model0]
    type = SlopeSaturationVoceIsotropicHardening
    saturated_hardening = 'state/R'
    initial_hardening_rate = 'state/theta0'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
