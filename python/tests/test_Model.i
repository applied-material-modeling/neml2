[Models]
  [foo]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'foo'
  []
  [bar]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'bar'
  []
  [baz]
    type = ScalarLinearCombination
    from = 'foo_residual bar_residual'
    to = 'foo_bar_residual'
  []
  [model]
    type = ComposedModel
    models = 'foo bar baz'
  []
[]
