[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    output_Scalar_names = 'aa'
    output_Scalar_values = '1.61753845'
  []
[]

[Models]
  [model0]
    type = ScalarParameterToVariable
    from = 1.61753845
    to = 'a'
  []
  [check_usage]
    type = ScalarLinearCombination
    from = 'a'
    to = 'aa'
  []
  [model]
    type = ComposedModel
    models = 'model0 check_usage'
  []
[]
