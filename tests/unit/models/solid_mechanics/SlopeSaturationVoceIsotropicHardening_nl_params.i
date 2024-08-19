[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    batch_shape = '(10)'
    input_scalar_names = 'state/internal/gamma_rate params/R params/theta0 state/internal/k'
    input_scalar_values = '0.1 100.0 110 50.0'
    output_scalar_names = 'state/internal/k_rate'
    output_scalar_values = '5.5'
    check_AD_first_derivatives = false
  []
[]

[Models]
  [R]
    type = ScalarInputParameter
    from = 'params/R'
  []
  [theta0]
    type = ScalarInputParameter
    from = 'params/theta0'
  []
  [model0]
    type = SlopeSaturationVoceIsotropicHardening
    saturated_hardening = 'R'
    initial_hardening_rate = 'theta0'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
