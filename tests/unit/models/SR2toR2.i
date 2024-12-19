[Tensors]
  [bar]
    type = FillR2
    values = '1 3 5 3 5 7 5 7 9'
  []
  [foo]
    type = FillSR2
    values = '1 5 9 7 5 3'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'state/sym'
    input_SR2_values = 'foo'
    output_R2_names = 'state/full'
    output_R2_values = 'bar'
    check_second_derivatives = true
  []
[]

[Models]
  [model]
    type = SR2toR2
    input = 'state/sym'
    output = 'state/full'
  []
[]
