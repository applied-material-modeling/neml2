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
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'E'
    force_SR2_values = 'strains'
    force_Scalar_names = 'temperature'
    force_Scalar_values = 'temperatures'
    save_as = 'result.pt'
  []
  [verification]
    type = VTestVerification
    driver = 'driver'
    SR2_names = 'output.stress'
    SR2_values = 'stresses'
    rtol = 1.0e-5
    atol = 1.0e-5
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
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 1000.0
  []
  [mu]
    type = ScalarLinearInterpolation
    argument = 'temperature'
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
    isotropic_hardening = 'isotropic_hardening'
  []
  [yield_zero]
    type = YieldFunction
    yield_stress = 0
    isotropic_hardening = 'isotropic_hardening'
    yield_function = 'fp_alt'
  []
  [flow]
    type = ComposedModel
    models = 'vonmises yield'
    automatic_nonlinear_parameter = false
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'yield_function'
    from = 'mandel_stress isotropic_hardening'
    to = 'flow_direction isotropic_hardening_direction'
  []
  [ri_flowrate]
    type = FBComplementarity
    a = 'yield_function'
    a_inequality = 'LE'
    b = 'gamma_rate_ri'
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
    yield_function = 'fp_alt'
    flow_rate = 'gamma_rate_rd'
  []
  [effective_strain_rate]
    type = SR2Invariant
    invariant_type = 'EFFECTIVE_STRAIN'
    tensor = 'E_rate'
    invariant = 'effective_strain_rate'
  []
  [g]
    type = KocksMeckingActivationEnergy
    strain_rate = 'effective_strain_rate'
    shear_modulus = 'mu'
    k = 1.38064e-20
    b = 2.48e-7
    eps0 = 1e10
  []
  [flowrate]
    type = KocksMeckingFlowSwitch
    g0 = 0.3708
    rate_independent_flow_rate = 'gamma_rate_ri'
    rate_dependent_flow_rate = 'gamma_rate_rd'
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
    variable = 'E'
  []
  [Eerate]
    type = SR2LinearCombination
    from = 'E_rate plastic_strain_rate'
    to = 'strain_rate'
    weights = '1 -1'
  []
  [modulus]
    type = ScalarLinearInterpolation
    argument = 'temperature'
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
    variable = 'stress'
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
  []
  [surface]
    type = ComposedModel
    models = "isoharden elasticity
              mandel_stress vonmises
              yield yield_zero normality eprate Eprate Erate Eerate
              ri_flowrate rd_flowrate g flowrate integrate_ep integrate_stress effective_strain_rate"
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'surface'
    unknowns = 'stress equivalent_plastic_strain gamma_rate_ri'
    residuals = 'stress_residual equivalent_plastic_strain_residual complementarity'
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
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
[]
