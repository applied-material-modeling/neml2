[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'foo_rate foo foo~1 t t~1'
    input_Scalar_values = '-0.3 1.1 0 1.3 1.1'
    output_Scalar_names = 'foo_residual'
    output_Scalar_values = '1.16'
  []
[]

[Models]
  [model]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'foo'
    time = 't'
  []
[]
