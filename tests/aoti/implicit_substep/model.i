# implicit_substep: the minimal implicit_simple ImplicitUpdate with adaptive
# substepping turned on (max_substepping_level = 2). The residual is a linear
# backward-Euler integration, so any single Newton solve converges -- the point
# here is to exercise the host-side substep DRIVER end-to-end through the
# compiled runtime: metadata role classification (x~1 -> old_state, t/t~1 ->
# cur/old force, x_rate -> static), driver dispatch, endpoint interpolation, and
# state chaining. Because the integration is exact-in-time, the substepped
# answer equals the single-shot answer (the "no-op equivalence" the MOOSE spike
# validated to ~7 figures). True convergence-recovery (a step that fails single
# shot but converges substepped) needs a nonlinear residual and is covered
# alongside the substepped-Jacobian tests.

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
    max_substepping_level = 2
  []
[]
