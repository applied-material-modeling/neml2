[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'A B c_A c_B c0'
    input_Scalar_values = '3 2 1 2 2'
    output_Scalar_names = 'C'
    output_Scalar_values = '9'
  []
[]

[Models]
  [model0]
    type = ScalarLinearCombination
    from = 'A B'
    to = 'C'
    weights = 'c_A c_B'
    weight_as_parameter = 'true true'
    offset = 'c0'
    offset_as_parameter = 'true'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
