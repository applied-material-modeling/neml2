# J2 power-law creep update built on the PowerLawCreepFlowRate leaf, reproducing
# the MOOSE ADPowerLawCreepStressUpdate law eps_dot_eff = A*<sigma_e>^n with
# A = 1e-30, n = 4 (Q = 0). Elasticity E = 80e9 Pa, nu = 0.33.
#
# Loading: uniaxial-style trace-free strain ramped to peak by t = 0.02 s and
# held to t = 0.1 s, so the deviatoric stress relaxes by creep during the hold.
[Tensors]
  [times]
    type = Python
    expr = 'Scalar(torch.linspace(0.0, 0.1, 50, dtype=torch.float64))'
  []
  [strains]
    type = Python
    expr = 'SR2(SR2.fill(2.0e-4, -1.0e-4, -1.0e-4, 0.0, 0.0, 0.0).data.unsqueeze(0) * torch.clamp(torch.linspace(0.0, 0.1, 50, dtype=torch.float64) / 0.02, max=1.0).reshape(50, 1))'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_SR2_names = 'E'
    prescribed_SR2_values = 'strains'
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
    invariant = 'yield_function'
  []
  [flow]
    type = ComposedModel
    models = 'vonmises'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'yield_function'
    from = 'mandel_stress'
    to = 'flow_direction'
  []
  [flow_rate]
    type = PowerLawCreepFlowRate
    coefficient = 1.0e-30
    exponent = 4
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
    coefficients = '80.0e9 0.33'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    rate_form = true
  []
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'stress'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'mandel_stress vonmises normality flow_rate Eprate Erate Eerate elasticity integrate_stress'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'stress'
    residuals = 'stress_residual'
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
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
[]
