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
  [copy1]
    type = CopyScalar
    from = 'state/foo'
    to = 'forces/foo'
  []
  [copy2]
    type = CopyScalar
    from = 'forces/foo'
    to = 'residual/foo'
  []
  [implicit]
    type = ComposedModel
    models = 'copy1 copy2'
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit'
    solver = 'newton'
  []
[]
