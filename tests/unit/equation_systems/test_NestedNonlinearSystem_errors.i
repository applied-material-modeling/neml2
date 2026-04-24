[Models]
  [rate]
    type = SampleRateModel
  []
  [integrate_foo]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'foo'
  []
  [integrate_bar]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'bar'
  []
  [integrate_baz]
    type = SR2BackwardEulerTimeIntegration
    variable = 'baz'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'rate integrate_foo integrate_bar integrate_baz'
  []
[]

[EquationSystems]
  [missing_variable]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'foo bar'
  []
  [nonexistent_variable]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'foo bar baz nonexistent'
  []
[]
