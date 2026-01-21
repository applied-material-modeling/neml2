[Tensors]
  [foo]
    type = FullScalar
    value = '1.0'
    batch_shape = '(5,2)'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'old_state/foo state/foo forces/t old_forces/t'
    input_Scalar_values = '0 foo 1.3 1.1'
    output_Scalar_names = 'state/foo'
    output_Scalar_values = '0'
  []
[]

[Models]
  [foo_rate]
    type = CopyScalar
    from = 'state/foo'
    to = 'state/foo_rate'
  []
  [integrate_foo]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/foo'
  []
  [implicit_model]
    type = ComposedModel
    models = 'foo_rate integrate_foo'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_model'
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
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
[]
