[Models]
  [rate]
    type = SampleRateModel
    temperature = 'T'
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
  [two_groups]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'foo bar; baz'
  []
  [three_groups]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'foo; bar; baz'
  []
  [single_group]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'foo bar baz'
  []
  [reordered]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'baz; bar foo'
  []
[]
