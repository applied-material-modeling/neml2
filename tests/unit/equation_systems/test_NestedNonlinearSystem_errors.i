[Models]
  [rate]
    type = SampleRateModel
  []
  [integrate_foo]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/foo'
  []
  [integrate_bar]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/bar'
  []
  [integrate_baz]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/baz'
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
    unknown_groups = 'state/foo state/bar'
    residual_groups = 'residual/foo residual/bar residual/baz'
  []
  [nonexistent_variable]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknown_groups = 'state/foo state/bar state/baz state/nonexistent'
    residual_groups = 'residual/foo residual/bar residual/baz'
  []
[]
