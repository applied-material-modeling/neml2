[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = "state/foo state/foo_rate old_state/foo
                          state/bar state/bar_rate old_state/bar
                          state/baz state/baz_rate old_state/baz
                          forces/t old_forces/t"
    input_scalar_values = "2 5 0
                           -1 -3 0
                           3 8 1
                           1.3 1.1"
    output_scalar_names = 'residual/foo_bar_baz'
    output_scalar_values = '1.0'
  []
[]

[Models]
  [integrate_foo]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/foo'
  []
  [integrate_bar]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/bar'
  []
  [residual_sum1]
    type = ScalarLinearCombination
    from_var = 'residual/foo residual/bar'
    to_var = 'residual/foo_bar'
  []
  [model1]
    type = ComposedModel
    models = 'integrate_foo integrate_bar residual_sum1'
  []
  [integrate_baz]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/baz'
  []
  [residual_sum2]
    type = ScalarLinearCombination
    from_var = 'residual/foo_bar residual/baz'
    to_var = 'residual/foo_bar_baz'
  []
  [model]
    type = ComposedModel
    models = 'model1 integrate_baz residual_sum2'
  []
[]
