# neml2
# Native port of tests/regression/solid_mechanics/mixed_control/model.i.
# Mixed strain/stress control via MixedControlSetup with a (100, 20) batch:
# 20 logspaced end times, 100 linear time steps. Component 0 is strain-
# controlled to 0.1; component 3 is stress-controlled to -130. SR2.fill 6-arg
# applies sqrt(2) Mandel scaling on shear slots (3,4,5).
[Tensors]
  # end_time = LogspaceScalar(-1, 5, 20) -> shape (20,)
  [end_time]
    type = Python
    expr = 'Scalar(torch.logspace(-1.0, 5.0, 20, dtype=torch.float64))'
  []
  # times = LinspaceScalar(0, end_time, 100) -> shape (100, 20)
  [times]
    type = Python
    expr = 'Scalar(end_time.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1))'
  []
  # max_conds = FillSR2(applied_strain=0.1, 0, 0, applied_stress=-130, 0, 0) batched (20,)
  # SR2.fill 6-arg keeps slots 0..2 verbatim and scales slots 3..5 by sqrt(2).
  # Here slots 3..5 are -130, 0, 0 -> -130*sqrt(2), 0, 0 in Mandel storage.
  [max_conds]
    type = Python
    expr = 'SR2(torch.tensor([0.1, 0.0, 0.0, -130.0 * (2.0 ** 0.5), 0.0, 0.0], dtype=torch.float64).unsqueeze(0).expand(20, 6).contiguous())'
  []
  # conditions = LinspaceSR2(0, max_conds, 100) -> shape (100, 20, 6)
  [conditions]
    type = Python
    expr = 'SR2(max_conds.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).reshape(100, 1, 1))'
  []
  # control = FillSR2(0, 1, 1, 1, 0, 0) batched (100, 20).
  # SR2.fill 6-arg scales slots 3..5 by sqrt(2): slot 3 = 1 -> sqrt(2).
  [control]
    type = Python
    expr = 'SR2(torch.tensor([0.0, 1.0, 1.0, 2.0 ** 0.5, 0.0, 0.0], dtype=torch.float64).reshape(1, 1, 6).expand(100, 20, 6).contiguous())'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_SR2_names = 'fixed_values control'
    prescribed_SR2_values = 'conditions control'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Models]
  [isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 50
    saturation_rate = 1.2
  []
  [kinharden]
    type = SR2LinearCombination
    from = 'X1 X2'
    to = 'X'
  []
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'stress'
  []
  [overstress]
    type = SR2LinearCombination
    from = 'mandel_stress X'
    to = 'O'
    weights = '1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'O'
    invariant = 'effective_stress'
  []
  [yield_surface]
    type = YieldFunction
    yield_stress = 10
    isotropic_hardening = 'isotropic_hardening'
  []
  [flow]
    type = ComposedModel
    models = 'overstress vonmises yield_surface'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'yield_function'
    from = 'mandel_stress isotropic_hardening'
    to = 'flow_direction isotropic_hardening_direction'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 155.22903539478642
    exponent = 4
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [X1rate]
    type = ChabochePlasticHardening
    back_stress = 'X1'
    C = 5000
    g = 8.246615467370033
    A = 1.224744871391589e-06
    a = 1.2
  []
  [X2rate]
    type = ChabochePlasticHardening
    back_stress = 'X2'
    C = 1000
    g = 4.245782220824175
    A = 1.224744871391589e-10
    a = 3.2
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [Erate]
    type = SR2VariableRate
    variable = 'strain'
  []
  [Eerate]
    type = SR2LinearCombination
    from = 'strain_rate plastic_strain_rate'
    to = 'elastic_strain_rate'
    weights = '1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
    rate_form = true
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
  []
  [integrate_X1]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X1'
  []
  [integrate_X2]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X2'
  []
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'stress'
  []
  [mixed]
    type = MixedControlSetup
    x_above = 'fixed_values'
    x_below = 'mixed_state'
    y = 'stress'
    z = 'strain'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'isoharden kinharden mandel_stress overstress vonmises yield_surface normality flow_rate eprate Eprate X1rate X2rate Erate Eerate elasticity integrate_stress integrate_ep integrate_X1 integrate_X2 mixed'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'mixed_state equivalent_plastic_strain X1 X2'
    residuals = 'stress_residual equivalent_plastic_strain_residual X1_residual X2_residual'
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
    unknowns_SR2 = 'mixed_state X1 X2'
    unknowns_Scalar = 'equivalent_plastic_strain'
  []
  [update]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model]
    type = ComposedModel
    models = 'update mixed'
    additional_outputs = 'mixed_state'
  []
[]
