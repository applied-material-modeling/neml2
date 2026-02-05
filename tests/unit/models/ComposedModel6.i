[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'ABC'
    input_Scalar_names = 'state/v'
    input_Scalar_values = '3'
    output_Scalar_names = 'residual/x residual/v'
    output_Scalar_values = '3 6'
  []
[]

[Models]
  [A]
    type = ScalarLinearCombination
    from_var = 'state/v'
    to_var = 'state/a'
    coefficients = '1'
  []
  [B]
    type = ScalarLinearCombination
    from_var = 'state/v'
    to_var = 'residual/x'
    coefficients = '1'
  []
  [C]
    type = ScalarLinearCombination
    from_var = 'state/a state/v'
    to_var = 'residual/v'
    coefficients = '1 1'
  []
  [BC]
    type = ComposedModel
    models = 'B C'
  []
  [ABC]
    type = ComposedModel
    models = 'A BC'
  []
[]
