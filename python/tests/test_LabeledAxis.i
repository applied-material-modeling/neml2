[Models]
  [foo]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/foo'
  []
  [bar]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/bar'
  []
  [baz]
    type = ScalarLinearCombination
    from_var = 'residual/foo residual/bar'
    to_var = 'residual/foo_bar'
  []
  [model]
    type = ComposedModel
    models = 'foo bar baz'
  []
[]
