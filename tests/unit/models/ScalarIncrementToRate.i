[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'forces/t old_forces/t forces/foo'
    input_Scalar_values = '1.3 1.1 1.2'
    output_Scalar_names = 'forces/foo_rate'
    output_Scalar_values = '6.0'
  []
[]

[Models]
  [model]
    type = ScalarIncrementToRate
    variable = 'forces/foo'
    rate = 'forces/foo_rate'
  []
[]
