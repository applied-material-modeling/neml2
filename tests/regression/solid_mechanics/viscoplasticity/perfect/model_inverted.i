# neml2
# model_inverted.i: identical physics to model.i, with the Perzyna flow rule stated as an
# implicit residual in INVERTED form (PerzynaPlasticFlowRateResidual) instead of
# as an explicit map (PerzynaPlasticFlowRate).
#
#     eta * gdot^(1/n) - <f> = 0      instead of      gdot = (<f>/eta)^n
#
# Same root, so this reuses model.i's gold reference unchanged -- that IS the
# test. What differs is what Newton sees: substituting the explicit map makes
# every residual degree-n in the stress, and Newton on a degree-n monomial
# converges only linearly, contracting by (1-1/n)^n. Carrying gdot as an unknown
# relocates that to a single 1/n power and leaves the other residuals affine in
# it. Measured on perfect: step-1 Newton iterations 15 -> 9, and 61 -> 55
# over the whole 99-step history.
#
# It also lifts the convergence basin by 10x. Driven six steps with no
# line search, the explicit form diverges once the per-step increment
# reaches 500x the parent scenario's; this form survives to 5000x.
[Tensors]
  [end_time]
    type = Python
    expr = 'Scalar(torch.logspace(-1.0, 5.0, 20, dtype=torch.float64))'
  []
  [times]
    type = Python
    expr = 'Scalar(end_time.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1))'
  []
  [max_strain]
    type = Python
    expr = 'SR2.fill(0.1, -0.05, -0.05, 0.0, 0.0, 0.0).dynamic_batch.expand(20)'
  []
  [strains]
    type = Python
    expr = 'SR2(max_strain.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).reshape(100, 1, 1))'
  []
[]

[Tensors]
  # Strictly positive seed for the flow-rate unknown. The inverted residual is
  # regularized at zero, but at step 1 there is no previous solve to extrapolate
  # from and a start at the origin costs a few extra iterations.
  [flow_rate_ic]
    type = Python
    expr = 'Scalar(1.0e-12)'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_SR2_names = 'E'
    prescribed_SR2_values = 'strains'
    ic_Scalar_names = 'flow_rate'
    ic_Scalar_values = 'flow_rate_ic'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Models]
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'stress'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'mandel_stress'
    invariant = 'effective_stress'
  []
  [yield_surface]
    type = YieldFunction
    yield_stress = 5
  []
  [flow]
    type = ComposedModel
    models = 'vonmises yield_surface'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'yield_function'
    from = 'mandel_stress'
    to = 'flow_direction'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRateResidual
    reference_stress = 100
    exponent = 2
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [Erate]
    type = SR2VariableRate
    variable = 'E'
  []
  [Eerate]
    type = SR2LinearCombination
    from = 'E_rate plastic_strain_rate'
    to = 'strain_rate'
    weights = '1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    rate_form = true
  []
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'stress'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'mandel_stress vonmises yield_surface normality flow_rate Eprate Erate Eerate elasticity integrate_stress'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'stress flow_rate'
    residuals = 'stress_residual flow_rate_residual'
  []
[]

[Solvers]
  [newton]
    type = Newton
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_SR2 = 'stress'
    unknowns_Scalar = 'flow_rate'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
[]
