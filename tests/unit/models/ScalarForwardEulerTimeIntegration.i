[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'state/foo_rate old_state/foo forces/t old_forces/t'
    input_scalar_values = '-0.3 0 1.3 1.1'
    output_scalar_names = 'state/foo'
    output_scalar_values = '-0.06'
  []
[]

[Models]
  [model]
    type = ScalarForwardEulerTimeIntegration
    variable = 'state/foo'
  []
[]
