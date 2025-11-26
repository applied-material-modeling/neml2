[Tensors]
  [times]
    type = ScalarVTestTimeSeries
    vtest = 'kocks_mecking.vtest'
    variable = 'time'
  []
  [strains]
    type = SR2VTestTimeSeries
    vtest = 'kocks_mecking.vtest'
    variable = 'strain'
  []
  [stresses]
    type = SR2VTestTimeSeries
    vtest = 'kocks_mecking.vtest'
    variable = 'stress'
  []
  [temperatures]
    type = ScalarVTestTimeSeries
    vtest = 'kocks_mecking.vtest'
    variable = 'temperature'
  []

  [T_controls]
    type = Scalar
    values = '750 850 950'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [E_values]
    type = Scalar
    values = '200000 175000 150000'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [mu_values]
    type = Scalar
    values = '76923.07692308 67307.69230769 57692.30769231'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
[]

[Drivers]
  [driver]
    type = SDTSolidMechanicsDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_strain = 'strains'
    prescribed_temperature = 'temperatures'
    save_as = 'result.pt'
  []
  [verification]
    type = VTestVerification
    driver = 'driver'
    SR2_names = 'output.state/S'
    SR2_values = 'stresses'
    rtol = 1.0e-5
    atol = 1.0e-5
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
    type = LinearIsotropicHardening
    hardening_modulus = 1000.0
  []
  [mu]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = 'T_controls'
    ordinate = 'mu_values'
  []
  [ys]
    type = KocksMeckingYieldStress
    shear_modulus = 'mu'
    C = -5.0486
  []
  [yield]
    type = YieldFunction
    yield_stress = 'ys'
    isotropic_hardening = 'state/internal/k'
  []
  [yield_zero]
    type = YieldFunction
    yield_stress = 0
    isotropic_hardening = 'state/internal/k'
    yield_function = 'state/internal/fp_alt'
  []
  [flow]
    type = ComposedModel
    models = 'vonmises yield'
    automatic_nonlinear_parameter = false
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'state/internal/fp'
    from = 'state/internal/M state/internal/k'
    to = 'state/internal/NM state/internal/Nk'
  []
  [ri_flowrate]
    type = RateIndependentPlasticFlowConstraint
    flow_rate = 'state/internal/gamma_rate_ri'
  []
  [km_sensitivity]
    type = KocksMeckingRateSensitivity
    A = -9.6187
    shear_modulus = 'mu'
    k = 1.38064e-20
    b = 2.48e-7
  []
  [km_viscosity]
    type = KocksMeckingFlowViscosity
    A = -9.6187
    B = -1.4819
    shear_modulus = 'mu'
    k = 1.38064e-20
    b = 2.48e-7
    eps0 = 1e10
  []
  [rd_flowrate]
    type = PerzynaPlasticFlowRate
    reference_stress = 'km_viscosity'
    exponent = 'km_sensitivity'
    yield_function = 'state/internal/fp_alt'
    flow_rate = 'state/internal/gamma_rate_rd'
  []
  [effective_strain_rate]
    type = SR2Invariant
    invariant_type = 'EFFECTIVE_STRAIN'
    tensor = 'forces/E_rate'
    invariant = 'forces/effective_strain_rate'
  []
  [g]
    type = KocksMeckingActivationEnergy
    activation_energy = 'state/g'
    shear_modulus = 'mu'
    k = 1.38064e-20
    b = 2.48e-7
    eps0 = 1e10
  []
  [flowrate]
    type = KocksMeckingFlowSwitch
    activation_energy = 'state/g'
    g0 = 0.3708
    rate_independent_flow_rate = 'state/internal/gamma_rate_ri'
    rate_dependent_flow_rate = 'state/internal/gamma_rate_rd'
    sharpness = 500.0
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
  [modulus]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = 'T_controls'
    ordinate = 'E_values'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = 'modulus 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
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
  [surface]
    type = ComposedModel
    models = "isoharden elasticity
              mandel_stress vonmises
              yield yield_zero normality eprate Eprate Erate Eerate
              ri_flowrate rd_flowrate g flowrate integrate_ep integrate_stress effective_strain_rate"
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'surface'
    solver = 'newton'
  []
[]
