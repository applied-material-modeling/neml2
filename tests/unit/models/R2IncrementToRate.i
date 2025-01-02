[Tensors]
  [foo]
    type = FillR2
    values = '1 2 3 4 5 6 7 8 9'
  []
  [bar]
    type = FillR2
    values = '5 10 15 20 25 30 35 40 45'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'forces/t old_forces/t'
    input_Scalar_values = '1.3 1.1'
    input_R2_names = 'forces/foo'
    input_R2_values = 'foo'
    output_R2_names = 'forces/foo_rate'
    output_R2_values = 'bar'
  []
[]

[Models]
  [model]
    type = R2IncrementToRate
    variable = 'forces/foo'
    rate = 'forces/foo_rate'
  []
[]
