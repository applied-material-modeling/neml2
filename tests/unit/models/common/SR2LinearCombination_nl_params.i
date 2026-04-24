[Tensors]
  [foo]
    type = FillSR2
    values = '1 2 3 4 5 6'
  []
  [bar]
    type = FillSR2
    values = '-1 -4 7 -1 9 1'
  []
  [baz]
    type = FillSR2
    values = '3 10 -11 6 -13 4'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'c_A c_B'
    input_Scalar_values = '1.0 -2.0'
    input_SR2_names = 'A B'
    input_SR2_values = 'foo bar'
    output_SR2_names = 'C'
    output_SR2_values = 'baz'
  []
[]

[Models]
  [model0]
    type = SR2LinearCombination
    from = 'A B'
    to = 'C'
    weights = 'c_A c_B'
    weight_as_parameter = 'true true'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
