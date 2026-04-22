[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'A B c'
    input_Scalar_values = '3 2 1'
    output_Scalar_names = 'C'
    output_Scalar_values = '6'
  []
[]

[Models]
  [model0]
    type = ScalarLinearCombination
    from = 'A B'
    to = 'C'
    weights = 'c c'
    weight_as_parameter = 'true true'
    offset = 'c'
    offset_as_parameter = 'true'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
