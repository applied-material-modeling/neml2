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
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknown_groups = 'state/foo state/bar; state/baz'
    residual_groups = 'residual/foo residual/bar; residual/baz'
  []
[]

[Solvers]
  [newton]
    type = Newton
    abs_tol = 1e-10
    rel_tol = 1e-08
    max_its = 20
    linear_solver = 'schur'
  []
  [schur]
    type = SchurComplement
    primary_group = 0
    primary_linear_solver = 'lu'
    schur_group = 1
    schur_linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]
