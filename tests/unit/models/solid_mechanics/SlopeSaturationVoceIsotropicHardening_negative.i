[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/gamma_rate state/internal/k'
    input_Scalar_values = '0.1 50.0'
    output_Scalar_names = 'state/internal/k_rate'
    output_Scalar_values = '-16.5'
  []
[]

[Models]
  [model]
    type = SlopeSaturationVoceIsotropicHardening
    saturated_hardening = -100
    initial_hardening_rate = 110.0
  []
[]
