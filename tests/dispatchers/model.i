[Models]
  [integrate_foo]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/foo'
  []
  [integrate_bar]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/bar'
  []
  [residual_sum]
    type = ScalarLinearCombination
    from_var = 'residual/foo residual/bar'
    to_var = 'residual/foo_bar'
  []
  [model]
    type = ComposedModel
    models = 'integrate_foo integrate_bar residual_sum'
  []
[]
