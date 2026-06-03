# forward_implicit: a composition of a forward leaf and an ImplicitUpdate.
# The forward leaf (MacaulaySplit) computes a rate from a structural input;
# the ImplicitUpdate integrates that rate. Exercises the segment partitioner:
# the exporter should emit one forward segment + one implicit segment.

[Models]
  [rate]
    # MacaulaySplit: x -> (max(x,0), max(-x,0)). The positive branch is what
    # the integrator consumes; the negative branch is an unused output we
    # rename so it doesn't collide with anything else in the state map.
    type = MacaulaySplit
    from = 'x'
    to_positive = 'y_rate'
    to_negative = 'y_neg_unused'
  []
  [integrate]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'y'
    time = 't'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'rate integrate'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'y'
    residuals = 'y_residual'
  []
[]

[Solvers]
  [newton]
    type = Newton
    abs_tol = 1e-10
    rel_tol = 1e-08
    max_its = 25
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
