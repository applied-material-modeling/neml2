[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'old_state/foo old_state/bar forces/temperature forces/t old_forces/t'
    input_scalar_values = '0 0 15 1.3 1.1'
    input_symr2_names = 'old_state/baz'
    input_symr2_values = '0'
    output_scalar_names = 'state/foo state/bar'
    output_scalar_values = '-1.43918 -2.55098'
    output_symr2_names = 'state/baz'
    output_symr2_values = '0'
  []
[]

[Solvers]
  [newton]
    type = Newton
    abs_tol = 1e-10
    rel_tol = 1e-08
    max_its = 20
  []
[]

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
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
[]
