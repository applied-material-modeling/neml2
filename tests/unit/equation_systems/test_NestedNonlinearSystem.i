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
  [two_groups]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'state/foo state/bar; state/baz'
    residuals = 'residual/foo residual/bar; residual/baz'
  []
  [three_groups]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'state/foo; state/bar; state/baz'
    residuals = 'residual/foo; residual/bar; residual/baz'
  []
  [single_group]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'state/foo state/bar state/baz'
    residuals = 'residual/foo residual/bar residual/baz'
  []
  [reordered]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'state/baz; state/bar state/foo'
    residuals = 'residual/baz; residual/bar residual/foo'
  []
[]
