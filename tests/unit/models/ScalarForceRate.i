[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'forces/foo old_forces/foo forces/t old_forces/t'
    input_scalar_values = '-0.3 0 1.3 1.1'
    output_scalar_names = 'forces/foo_rate'
    output_scalar_values = '-1.5'
  []
[]

[Models]
  [model]
    type = ScalarVariableRate
    variable = 'forces/foo'
    rate = 'forces/foo_rate'
  []
[]
