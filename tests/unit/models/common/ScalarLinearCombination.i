[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'A B'
    input_Scalar_values = '3 2'
    output_Scalar_names = 'C'
    output_Scalar_values = '7'
  []
[]

[Models]
  [model]
    type = ScalarLinearCombination
    from = 'A B'
    to = 'C'
    offset = 2.0
  []
[]
