[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'foo~1 bar~1 T t t~1'
    input_Scalar_values = '0 0 15 1.3 1.1'
    input_SR2_names = 'baz~1'
    input_SR2_values = '0'
    output_Scalar_names = 'sum'
    output_Scalar_values = '-3.9902'
  []
[]

[Models]
  [rate]
    type = SampleRateModel
    temperature = 'T'
  []
  [integrate_foo]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'foo'
    time = 't'
  []
  [integrate_bar]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'bar'
    time = 't'
  []
  [integrate_baz]
    type = SR2BackwardEulerTimeIntegration
    variable = 'baz'
    time = 't'
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
    unknowns = 'foo bar baz'
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
  [implicit_update]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
  [baz_tr]
    type = SR2Invariant
    tensor = 'baz'
    invariant = 'baz_tr'
    invariant_type = 'I1'
  []
  [sum]
    type = ScalarLinearCombination
    from = 'foo bar baz_tr'
    to = 'sum'
  []
  [model]
    type = ComposedModel
    models = 'implicit_update baz_tr sum'
  []
[]
