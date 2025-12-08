[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'old_state/foo old_state/bar forces/temperature forces/t old_forces/t'
    input_Scalar_values = '0 0 15 1.3 1.1'
    input_SR2_names = 'old_state/baz'
    input_SR2_values = '0'
    output_Scalar_names = 'state/sum'
    output_Scalar_values = '-3.9902'
  []
[]

[Solvers]
  [newton]
    type = Newton
    abs_tol = 1e-10
    rel_tol = 1e-08
    max_its = 100
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
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
  [implicit_update]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
  [baz_tr]
    type = SR2Invariant
    tensor = 'state/baz'
    invariant = 'state/baz_tr'
    invariant_type = 'I1'
  []
  [sum]
    type = ScalarLinearCombination
    from_var = 'state/foo state/bar state/baz_tr'
    to_var = 'state/sum'
  []
  [model]
    type = ComposedModel
    models = 'implicit_update baz_tr sum'
  []
[]
