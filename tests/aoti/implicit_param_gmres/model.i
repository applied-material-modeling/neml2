# neml2
# implicit_param_gmres: the same single-unknown ImplicitUpdate as implicit_param
# (promotable scalar weight `w` = rate.weight_0 inside a backward-Euler residual),
# but with a MATRIX-FREE GMRES `param_sensitivity_solver`. Compiled with a promoted
# `w` + `-d x:...weight_0`, this exercises the iterative PARAMETER-sensitivity
# (ParamIFT) path: the artifact emits `_dr_dparam` but NO `_solve_param` -- the C++
# runtime runs a Krylov solve over the dense A to form du/dtheta. Validated against
# the direct DenseLU solve of the identical model (implicit_param).

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
  [gmres]
    type = GMRES
  []
[]

[Models]
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    param_sensitivity_solver = 'gmres'
  []
[]
