[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    output_SR2_names = 'E'
    output_SR2_values = 'T'
  []
[]

[Models]
  [E]
    type = SR2ConstantParameter
    value = 'T'
  []
[]

[Tensors]
  [T]
    type = FillSR2
    values = '-1 -4 7 -1 9 1'
  []
[]
