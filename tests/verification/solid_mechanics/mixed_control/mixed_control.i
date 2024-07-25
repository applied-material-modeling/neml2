[Tensors]
  [times]
    type = VTestTimeSeries
    vtest = 'mixed_control.vtest'
    variable = 'time'
    variable_type = 'SCALAR'
  []
  [strains]
    type = VTestTimeSeries
    vtest = 'mixed_control.vtest'
    variable = 'strain'
    variable_type = 'SYMR2'
  []
  [stresses]
    type = VTestTimeSeries
    vtest = 'mixed_control.vtest'
    variable = 'stress'
    variable_type = 'SYMR2'
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
    type = SolidMechanicsDriver
    model = 'model_with_output'
    control = 'MIXED'
    times = 'times'
    prescribed_mixed_conditions = 'stresses'
    prescribed_control = 'control'
  []
  [verification]
    type = VTestVerification
    driver = 'driver'
    variables = 'output.state/S output.state/E'
    references = 'stresses strains'
    atol = 1e-5
    rtol = 1e-5
  []
[]

[Solvers]
  [newton]
    type = Newton
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
    from_var = 'state/internal/X1 state/internal/X2'
    to_var = 'state/internal/X'
  []
  [mandel_stress]
    type = IsotropicMandelStress
  []
  [overstress]
    type = SR2LinearCombination
    to_var = 'state/internal/O'
    from_var = 'state/internal/M state/internal/X'
    coefficients = '1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/O'
    invariant = 'state/internal/s'
  []
  [yield]
    type = YieldFunction
    yield_stress = 10
    isotropic_hardening = 'state/internal/k'
  []
  [flow]
    type = ComposedModel
    models = 'overstress vonmises yield'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'state/internal/fp'
    from = 'state/internal/M state/internal/k'
    to = 'state/internal/NM state/internal/Nk'
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
    back_stress = 'state/internal/X1'
    C = 5000
    g = 8.246615467370033 # 10.1 * sqrt(2/3)
    A = 1.224744871391589e-06 # 1.0e-6 * sqrt(3/2)
    a = 1.2
  []
  [X2rate]
    type = ChabochePlasticHardening
    back_stress = 'state/internal/X2'
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
    variable = 'state/E'
    rate = 'state/E_rate'
  []
  [Eerate]
    type = SR2LinearCombination
    from_var = 'state/E_rate state/internal/Ep_rate'
    to_var = 'state/internal/Ee_rate'
    coefficients = '1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    youngs_modulus = 1e5
    poisson_ratio = 0.3
    rate_form = true
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/internal/ep'
  []
  [integrate_X1]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/internal/X1'
  []
  [integrate_X2]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/internal/X2'
  []
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/S'
  []
  [mixed]
    type = MixedControlSetup
  []
  [rename]
    type = CopySR2
    from = 'residual/S'
    to = 'residual/mixed_state'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'isoharden kinharden mandel_stress overstress vonmises yield normality flow_rate eprate Eprate X1rate X2rate Erate Eerate elasticity integrate_stress integrate_ep integrate_X1 integrate_X2 mixed rename'
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
  [model_with_output]
    type = ComposedModel
    models = 'model mixed'
    additional_outputs = 'state/mixed_state'
  []
[]

