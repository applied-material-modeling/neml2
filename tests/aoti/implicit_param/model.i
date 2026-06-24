# implicit_param: a minimal single-unknown ImplicitUpdate carrying a promotable
# scalar parameter inside its residual -- the smallest model that exercises the
# implicit AOTI parameter-derivative path (du/dtheta = -A^-1 dr/dtheta).
#
# The residual is a backward-Euler step  r = x - x~1 - dt * x_rate  with the
# rate  x_rate = w * x  produced by a ScalarLinearCombination whose single
# weight `w` is promotable. Hence  r = x(1 - dt*w) - x~1, solved for x:
#   x = x~1 / (1 - dt*w).
# The parameter sensitivity is solution-dependent:
#   dr/dw = -dt*x,  du/dw = dt*x / (1 - dt*w),
# so it pins both the analytic A = dr/dx (= 1 - dt*w, from the chain rule) and
# the reverse-mode dr/dw (which reads the converged x) against finite diffs.

[Models]
  [rate]
    type = ScalarLinearCombination
    from = 'x'
    to = 'x_rate'
    weights = '-2'
  []
  [integrate]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'x'
    time = 't'
  []
  [residual]
    type = ComposedModel
    models = 'rate integrate'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'residual'
    unknowns = 'x'
    residuals = 'x_residual'
  []
[]

[Solvers]
  [newton]
    type = Newton
    abs_tol = 1e-12
    rel_tol = 1e-10
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
