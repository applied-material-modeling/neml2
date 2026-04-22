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
    input_Scalar_names = 'foo~1 foo t t~1'
    input_Scalar_values = '0 foo 1.3 1.1'
    output_Scalar_names = 'foo'
    output_Scalar_values = '0'
  []
[]

[Models]
  [foo_rate]
    type = CopyScalar
    from = 'foo'
    to = 'foo_rate'
  []
  [integrate_foo]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'foo'
    time = 't'
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
    unknowns = 'foo'
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
