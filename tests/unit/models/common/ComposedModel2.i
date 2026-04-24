[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = "foo foo_rate foo~1
                          bar bar_rate bar~1
                          baz baz_rate baz~1
                          t t~1"
    input_Scalar_values = "2 5 0
                           -1 -3 0
                           3 8 1
                           1.3 1.1"
    output_Scalar_names = 'foo_bar_baz_residual'
    output_Scalar_values = '1.0'
  []
[]

[Models]
  [integrate_foo]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'foo'
    time = 't'
  []
  [integrate_bar]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'bar'
    time = 't'
  []
  [residual_sum1]
    type = ScalarLinearCombination
    from = 'foo_residual bar_residual'
    to = 'foo_bar_residual'
  []
  [model1]
    type = ComposedModel
    models = 'integrate_foo integrate_bar residual_sum1'
  []
  [integrate_baz]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'baz'
    time = 't'
  []
  [residual_sum2]
    type = ScalarLinearCombination
    from = 'foo_bar_residual baz_residual'
    to = 'foo_bar_baz_residual'
  []
  [model]
    type = ComposedModel
    models = 'model1 integrate_baz residual_sum2'
  []
[]
