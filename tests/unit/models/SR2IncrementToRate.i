[Tensors]
  [foo]
    type = FillSR2
    values = '1 2 3'
  []
  [bar]
    type = FillSR2
    values = '5 10 15'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'forces/t old_forces/t'
    input_Scalar_values = '1.3 1.1'
    input_SR2_names = 'forces/foo'
    input_SR2_values = 'foo'
    output_SR2_names = 'forces/foo_rate'
    output_SR2_values = 'bar'
  []
[]

[Models]
  [model]
    type = SR2IncrementToRate
    variable = 'forces/foo'
    rate = 'forces/foo_rate'
  []
[]
