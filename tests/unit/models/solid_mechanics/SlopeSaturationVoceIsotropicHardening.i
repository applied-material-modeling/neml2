[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'state/internal/gamma_rate state/internal/k'
    input_scalar_values = '0.1 50.0'
    output_scalar_names = 'state/internal/k_rate'
    output_scalar_values = '5.5'
  []
[]

[Models]
  [model]
    type = SlopeSaturationVoceIsotropicHardening
    saturated_hardening = 100
    initial_hardening_rate = 110.0
  []
[]
