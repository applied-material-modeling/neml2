[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/A state/B state/C state/D'
    input_Scalar_values = '3.1 2.5 4.6 7.8'
    output_Scalar_names = 'state/E'
    output_Scalar_values = '417.105'
  []
[]

[Models]
  [model]
    type = ScalarMultiplication
    from_var = 'state/A state/B state/C state/D'
    to_var = 'state/E'
    coefficient = 1.5
  []
[]
