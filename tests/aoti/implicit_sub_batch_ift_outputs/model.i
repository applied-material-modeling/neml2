# Per-site backward Euler dw/dt = v u - w for a three-site w, then outputs that read
# the sub-batched u and the plain v next to the solved w:
#   p = v / u (site-local, independent of w), a = w u, c = sum_i w_i.
# p:u stays "paired" although p:w is a zero pair after a solve, and c:v (plain)
# works although the post-solve segment also takes the sub-batched u.

[Settings]
  [example_batch_shape]
    u = '(2; 3)'
    w = '(2; 3)'
  []
[]

[Models]
  [vu]
    type = ScalarMultiplication
    from = 'v u'
    to = 'vu'
  []
  [rate]
    type = ScalarLinearCombination
    from = 'vu w'
    to = 'w_rate'
    weights = '1 -1'
  []
  [integrate]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'w'
    time = 't'
  []
  [residual]
    type = ComposedModel
    models = 'vu rate integrate'
  []
  [update]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
  [p]
    type = ScalarMultiplication
    from = 'v u'
    to = 'p'
    reciprocal = 'false true'
  []
  [a]
    type = ScalarMultiplication
    from = 'w u'
    to = 'a'
  []
  [c]
    type = ScalarIntermediateSum
    from = 'w'
    to = 'c'
  []
  [model]
    type = ComposedModel
    models = 'update p a c'
    additional_outputs = 'w'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'residual'
    unknowns = 'w'
    residuals = 'w_residual'
  []
[]

[Solvers]
  [newton]
    type = Newton
    abs_tol = 1e-12
    rel_tol = 1e-10
    max_its = 10
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]
