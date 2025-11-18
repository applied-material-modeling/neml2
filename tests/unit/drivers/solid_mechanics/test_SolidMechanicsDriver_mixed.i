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
    nstep = 10
  []
  [applied_strain]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.001
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
    batch_shape = '(10,20)'
    value = 0.0
  []
  [one_control]
    type = FullScalar
    batch_shape = '(10,20)'
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
    nstep = 10
  []
  [control]
    type = FillSR2
    values = 'zero_control one_control one_control one_control zero_control zero_control'
  []
[]

[Drivers]
  [driver]
    type = SDTSolidMechanicsDriver
    model = 'model'
    control = 'MIXED'
    prescribed_time = 'times'
    prescribed_mixed_driving_force = 'conditions'
    prescribed_mixed_control_signal = 'control'
    save_as = 'drivers/solid_mechanics/test_SolidMechanicsDriver_mixed.pt'
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
  []
  [Eerate]
    type = SR2LinearCombination
    from_var = 'state/E_rate state/internal/Ep_rate'
    to_var = 'state/internal/Ee_rate'
    coefficients = '1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
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
    above_variable = 'state/S'
    below_variable = 'state/E'
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
  [model0]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
  [model]
    type = ComposedModel
    models = 'model0 mixed'
    additional_outputs = 'state/mixed_state'
  []
[]

