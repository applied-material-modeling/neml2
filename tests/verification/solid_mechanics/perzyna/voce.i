[Tensors]
  [times]
    type = VTestTimeSeries
    vtest = 'voce.vtest'
    variable = 'time'
    variable_type = 'SCALAR'
  []
  [strains]
    type = VTestTimeSeries
    vtest = 'voce.vtest'
    variable = 'strain'
    variable_type = 'SYMR2'
  []
  [stresses]
    type = VTestTimeSeries
    vtest = 'voce.vtest'
    variable = 'stress'
    variable_type = 'SYMR2'
  []
[]

[Drivers]
  [driver]
    type = SolidMechanicsDriver
    model = 'model'
    times = 'times'
    prescribed_strains = 'strains'
  []
  [verification]
    type = VTestVerification
    driver = 'driver'
    variables = 'output.state/S'
    references = 'stresses'
    atol = 1e-5
    rtol = 1e-8
  []
[]

[Solvers]
  [newton]
    type = Newton
  []
[]

[Models]
  [mandel_stress]
    type = IsotropicMandelStress
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/M'
    invariant = 'state/internal/s'
  []
  [isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 100
    saturation_rate = 1.1
  []
  [yield]
    type = YieldFunction
    yield_stress = 10
    isotropic_hardening = 'state/internal/k'
  []
  [flow]
    type = ComposedModel
    models = 'vonmises yield'
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
    reference_stress = 500
    exponent = 5
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [Erate]
    type = SR2VariableRate
    variable = 'forces/E'
    rate = 'forces/E_rate'
  []
  [Eerate]
    type = SR2LinearCombination
    from_var = 'forces/E_rate state/internal/Ep_rate'
    to_var = 'state/internal/Ee_rate'
    coefficients = '1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    youngs_modulus = 124000
    poisson_ratio = 0.32
    rate_form = true
  []
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/S'
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/internal/ep'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'mandel_stress vonmises isoharden yield normality flow_rate Eprate eprate Erate Eerate elasticity integrate_stress integrate_ep'
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
[]
