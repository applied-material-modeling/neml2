[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'foo foo~1 t t~1'
    input_Scalar_values = '-0.3 0 1.3 1.1'
    output_Scalar_names = 'foo_rate'
    output_Scalar_values = '-1.5'
  []
[]

[Models]
  [model]
    type = ScalarVariableRate
    variable = 'foo'
    time = 't'
  []
[]
