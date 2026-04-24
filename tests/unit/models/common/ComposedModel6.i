[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'ABC'
    input_Scalar_names = 'v'
    input_Scalar_values = '3'
    output_Scalar_names = 'x_residual v_residual'
    output_Scalar_values = '3 6'
  []
[]

[Models]
  [A]
    type = CopyScalar
    from = 'v'
    to = 'a'
  []
  [B]
    type = CopyScalar
    from = 'v'
    to = 'x_residual'
  []
  [C]
    type = ScalarLinearCombination
    from = 'a v'
    to = 'v_residual'
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
