[Tensors]
  [times]
    type = ScalarVTestTimeSeries
    vtest = 'mixed_control.vtest'
    variable = 'time'
  []
  [strains]
    type = SR2VTestTimeSeries
    vtest = 'mixed_control.vtest'
    variable = 'strain'
  []
  [stresses]
    type = SR2VTestTimeSeries
    vtest = 'mixed_control.vtest'
    variable = 'stress'
  []
  [one_control]
    type = FullScalar
    batch_shape = '(100,1)'
    value = 1.0
  []
  [zero_control]
    type = FullScalar
    batch_shape = '(100,1)'
    value = 0.0
  []
  [control]
    type = FillSR2
    values = 'one_control one_control one_control zero_control one_control one_control'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'stresses_force control'
    force_SR2_values = 'stresses control'
    save_as = 'result.pt'
  []
  [verification]
    type = VTestVerification
    driver = 'driver'
    SR2_names = 'output.stress output.E'
    SR2_values = 'stresses strains'
    atol = 1e-5
    rtol = 1e-5
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
    to = 'O'
    from = 'mandel_stress X'
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
    from = 'mandel_stress isotropic_hardening X'
    to = 'flow_direction isotropic_hardening_direction kinematic_hardening_direction'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 155.22903539478642 # 200 * (2/3)^(5/8)
    exponent = 4
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [X1rate]
    type = ChabochePlasticHardening
    back_stress = 'X1'
    C = 5000
    g = 8.246615467370033 # 10.1 * sqrt(2/3)
    A = 1.224744871391589e-06 # 1.0e-6 * sqrt(3/2)
    a = 1.2
  []
  [X2rate]
    type = ChabochePlasticHardening
    back_stress = 'X2'
    C = 1000
    g = 4.245782220824175 # 5.2 * sqrt(2/3)
    A = 1.224744871391589e-10 # 1.0e-10 * sqrt(3/2)
    a = 3.2
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
    x_above = 'stress'
    x_below = 'E'
  []
  [y_constraint]
    type = SR2LinearCombination
    from = 'y stresses_force'
    to = 'y_residual'
    weights = '1 -1'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'isoharden kinharden mandel_stress overstress vonmises yield normality flow_rate eprate Eprate X1rate X2rate Erate Eerate elasticity integrate_stress integrate_ep integrate_X1 integrate_X2 mixed y_constraint'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'stress E equivalent_plastic_strain X1 X2'
    residuals = 'stress_residual y_residual equivalent_plastic_strain_residual X1_residual X2_residual'
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
    unknowns_SR2 = 'stress E X1 X2'
    unknowns_Scalar = 'equivalent_plastic_strain'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
[]
