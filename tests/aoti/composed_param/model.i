# composed_param: forward -> implicit -> forward, with a promotable scalar
# parameter in EACH segment, to exercise the multi-segment AOTI parameter
# Jacobian carrier (du/dθ composed across segments). Plain batch.
#
#   seg0 (forward):  x_rate_ext = c * f                       (param c)
#   seg1 (implicit): solve x from r = x - x~1 - dt*(m*x_rate_ext)  (param m)
#                    => x = x~1 + dt*m*x_rate_ext
#   seg2 (forward):  y = b * x                                (param b)
#
#   y = b * (x~1 + dt*m*c*f), dt = t - t~1, so the closed-form sensitivities are
#     d(y)/dc = b*dt*m*f      (seg0 param, propagates through implicit + seg2)
#     d(y)/dm = b*dt*c*f      (implicit param, ParamIFT direct + seg2 jvp)
#     d(y)/db = x             (seg2 param, direct)

[Models]
  [scale]
    type = ScalarLinearCombination
    from = 'f'
    to = 'x_rate_ext'
    weights = '0.7'
  []
  [modrate]
    type = ScalarLinearCombination
    from = 'x_rate_ext'
    to = 'x_rate'
    weights = '1.3'
  []
  [intg]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'x'
    time = 't'
  []
  [impl_residual]
    type = ComposedModel
    models = 'modrate intg'
  []
  [out]
    type = ScalarLinearCombination
    from = 'x'
    to = 'y'
    weights = '2.0'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'impl_residual'
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
  [return_map]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
  [model]
    type = ComposedModel
    models = 'scale return_map out'
  []
[]
