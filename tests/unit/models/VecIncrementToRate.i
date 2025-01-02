[Tensors]
  [foo]
    type = Vec
    values = '1 2 3'
  []
  [bar]
    type = Vec
    values = '5 10 15'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'forces/t old_forces/t'
    input_Scalar_values = '1.3 1.1'
    input_Vec_names = 'forces/foo'
    input_Vec_values = 'foo'
    output_Vec_names = 'forces/foo_rate'
    output_Vec_values = 'bar'
  []
[]

[Models]
  [model]
    type = VecIncrementToRate
    variable = 'forces/foo'
    rate = 'forces/foo_rate'
  []
[]
