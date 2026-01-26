[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    output_Scalar_names = 'state/aa'
    output_Scalar_values = '1.61753845'
  []
[]

[Models]
  [model0]
    type = ScalarParameterToState
    from = 1.61753845
    to = 'state/a'
  []
  [check_usage]
    type = ScalarLinearCombination
    coefficients = '1.0'
    from_var = 'state/a'
    to_var = 'state/aa'
  []
  [model]
    type = ComposedModel
    models = 'model0 check_usage'
  []
[]
