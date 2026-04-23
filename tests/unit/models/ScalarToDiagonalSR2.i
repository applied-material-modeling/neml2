[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'input'
    input_Scalar_values = 'input'
    output_SR2_names = 'output'
    output_SR2_values = 'output'
    check_second_derivatives = true
  []
[]

[Tensors]
  [input]
    type = Scalar
    values = '2.1'
  []
  [output]
    type = FillSR2
    values = '2.1 2.1 2.1 0.0 0.0 0.0'
  []
[]

[Models]
  [model]
    type = ScalarToDiagonalSR2
    input = 'input'
    output = 'output'
  []
[]
