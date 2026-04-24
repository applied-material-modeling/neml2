[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'foo_rate t foo~1 t~1'
    input_Scalar_values = '-0.3 1.3 0 1.1'
    output_Scalar_names = 'foo'
    output_Scalar_values = '-0.06'
  []
[]

[Models]
  [model]
    type = ScalarForwardEulerTimeIntegration
    variable = 'foo'
    time = 't'
  []
[]
