[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'old_state/foo old_state/bar forces/temperature forces/t old_forces/t'
    input_Scalar_values = '0 0 15 1.3 1.1'
    input_SR2_names = 'old_state/baz'
    input_SR2_values = '0'
    output_Scalar_names = 'state/foo state/bar'
    output_Scalar_values = '-1.43918 -2.55098'
    output_SR2_names = 'state/baz'
    output_SR2_values = '0'
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
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
  []
[]

[Solvers]
  [newton]
    type = Newton
    abs_tol = 1e-10
    rel_tol = 1e-08
    max_its = 20
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
[]
