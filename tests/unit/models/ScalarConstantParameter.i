[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    batch_shape = '(10)'
    output_scalar_names = 'parameters/E'
    output_scalar_values = 'T'
    check_first_derivatives = false
    check_second_derivatives = false
  []
[]

[Models]
  [E]
    type = ScalarConstantParameter
    value = 'T'
  []
[]

[Tensors]
  [T]
    type = FullScalar
    value = 20.0
  []
[]
