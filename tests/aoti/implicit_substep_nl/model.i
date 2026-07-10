# implicit_substep_nl: a minimal NONLINEAR scalar implicit with substepping on,
# used to validate the substepped consistent tangent across genuine bisection.
#
# The unknown x is fed as the Perzyna "yield function", so the rate is
# x_rate = (max(x,0)/eta)^n and the backward-Euler residual is
#   r(x) = x - x~1 - (t - t~1) * (max(x,0)/eta)^n,
# which is nonlinear in x (n > 1). A large enough increment makes the single
# Newton solve exceed max_its; substepping bisects the increment until each
# sub-step converges, and the accumulated Jacobian must match finite differences
# of the substepped forward.

[Models]
  [rate]
    type = PerzynaPlasticFlowRate
    yield_function = 'x'
    flow_rate = 'x_rate'
    reference_stress = 1.0
    exponent = 3
  []
  [integrate]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'x'
    time = 't'
  []
  [residual_model]
    type = ComposedModel
    models = 'rate integrate'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'residual_model'
    unknowns = 'x'
    residuals = 'x_residual'
  []
[]

[Solvers]
  [newton]
    type = Newton
    abs_tol = 1e-10
    rel_tol = 1e-08
    # Generous cap: a hard dt (e.g. dt=8) still fails single-shot -- Newton
    # overshoots the cubic rate to a non-finite residual -- so the failure and
    # the substepped recovery are unambiguous (not a marginal max-iters edge)
    # and deterministic across routes.
    max_its = 25
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_Scalar = 'x'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
    max_substepping_level = 8
  []
[]
