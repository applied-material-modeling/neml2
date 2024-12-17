[Tensors]
  [foo]
    type = FillR2
    values = '1 2 3 4 5 6 7 8 9'
  []
  [bar]
    type = FillSR2
    values = '1 5 9 7 5 3'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_R2_names = 'state/full'
    input_R2_values = 'foo'
    output_SR2_names = 'state/notfull'
    output_SR2_values = 'bar'
    check_second_derivatives = true
  []
[]

[Models]
  [model]
    type = R2toSR2
    input = 'state/full'
    output = 'state/notfull'
  []
[]
