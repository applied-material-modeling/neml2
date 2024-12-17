[Tensors]
  [foo]
    type = FillR2
    values = '1 2 3 4 5 6 7 8 9'
  []
  [bar]
    type = FillWR2
    values = '1 -2 1'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_R2_names = 'state/full'
    input_R2_values = 'foo'
    output_WR2_names = 'state/notfull'
    output_WR2_values = 'bar'
    check_second_derivatives = true
  []
[]

[Models]
  [model]
    type = R2toWR2
    input = 'state/full'
    output = 'state/notfull'
  []
[]
