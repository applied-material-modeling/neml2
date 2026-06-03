# implicit_simple: minimal ImplicitUpdate. Just a backward-Euler time
# integration on a scalar variable -- a single-unknown, 1-D Newton solve that
# converges in one step. Exercises the rhs/step/ift/predictor path without
# touching LinearCombination (which has a separate pre-existing AOTI bug).

[Models]
  [integrate]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'x'
    time = 't'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'integrate'
    unknowns = 'x'
    residuals = 'x_residual'
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
