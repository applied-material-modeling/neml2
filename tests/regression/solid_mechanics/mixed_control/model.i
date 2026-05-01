[Tensors]
  [end_time]
    type = LogspaceScalar
    start = -1
    end = 5
    nstep = 20
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = end_time
    nstep = 100
  []
  [applied_strain]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.1
  []
  [applied_stress]
    type = FullScalar
    batch_shape = '(20)'
    value = -130
  []
  [zero]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.0
  []
  [zero_control]
    type = FullScalar
    batch_shape = '(100,20)'
    value = 0.0
  []
  [one_control]
    type = FullScalar
    batch_shape = '(100,20)'
    value = 1.0
  []
  [max_conds]
    type = FillSR2
    values = 'applied_strain zero zero applied_stress zero zero'
  []
  [conditions]
    type = LinspaceSR2
    start = 0
    end = max_conds
    nstep = 100
  []
  [control]
    type = FillSR2
    values = 'zero_control one_control one_control one_control zero_control zero_control'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'fixed_values control'
    force_SR2_values = 'conditions control'
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
  [yield]
    type = YieldFunction
    yield_stress = 10
    isotropic_hardening = 'isotropic_hardening'
  []
  [flow]
    type = ComposedModel
    models = 'overstress vonmises yield'
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
    models = 'isoharden kinharden mandel_stress overstress vonmises yield normality flow_rate eprate Eprate X1rate X2rate Erate Eerate elasticity integrate_stress integrate_ep integrate_X1 integrate_X2 mixed'
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
