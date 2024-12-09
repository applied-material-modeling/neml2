[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    output_scalar_names = 'parameters/E'
    output_scalar_values = 'T'
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
