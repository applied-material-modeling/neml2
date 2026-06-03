# neml2
[Tensors]
  # Ported from the C++ `type = Scalar` declarative form, which the native
  # factory does not (yet) support. `intermediate_dimension = 1` on the C++
  # side corresponds to `sub_batch_ndim = 1` on the native side and tags
  # the lookup-table axis the interpolator slides along.
  [T_controls]
    type = Python
    expr = 'Scalar(torch.linspace(300.0, 1200.0, 20, dtype=torch.float64)).with_sub_batch(1)'
  []
  [mu_values]
    type = Python
    expr = 'Scalar(torch.tensor([76670.48346056, 75465.18012589, 74314.80514263, 73374.72880675, 72651.54680595, 71928.36480514, 71120.75130575, 70035.97830454, 68951.20530333, 67842.26597027, 66399.97991161, 65315.20691041, 63884.85335476, 62763.98151868, 61373.80474086, 59927.44073925, 58481.07673765, 56544.43551627, 54599.93973483, 52791.98473282], dtype=torch.float64)).with_sub_batch(1)'
  []
  [E_test]
    type = Python
    expr = 'Scalar(torch.tensor(1.0e5, dtype=torch.float64))'
  []
[]

[Models]
  [A]
    type = ScalarConstantParameter
    value = -8.679
  []
  [B]
    type = ScalarConstantParameter
    value = -0.744
  []
  [C]
    type = ScalarConstantParameter
    value = -5.41
  []
  [g0]
    type = KocksMeckingIntercept
    A = 'A'
    B = 'B'
    C = 'C'
  []
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
    hardening_modulus = 1000
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
    C = 'C'
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
    A = 'A'
    shear_modulus = 'mu'
    k = 1.38064e-20
    b = 2.474e-7
  []
  [km_viscosity]
    type = KocksMeckingFlowViscosity
    A = 'A'
    B = 'B'
    shear_modulus = 'mu'
    k = 1.38064e-20
    b = 2.474e-7
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
    tensor = 'strain_rate'
    invariant = 'effective_strain_rate'
  []
  [g]
    type = KocksMeckingActivationEnergy
    strain_rate = 'effective_strain_rate'
    shear_modulus = 'mu'
    k = 1.38064e-20
    b = 2.474e-7
    eps0 = 1e10
  []
  [flowrate]
    type = KocksMeckingFlowSwitch
    activation_energy = 'activation_energy'
    g0 = 'g0'
    rate_independent_flow_rate = 'gamma_rate_ri'
    rate_dependent_flow_rate = 'gamma_rate_rd'
    sharpness = 100.0
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
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
    coefficients = 'E_test 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
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
  [mixed]
    type = MixedControlSetup
    x_above = 'mixed_control'
    x_below = 'mixed_state'
    y = 'stress'
    z = 'strain'
  []
  [mixed_old]
    type = MixedControlSetup
    control = 'control~1'
    x_above = 'mixed_control~1'
    x_below = 'mixed_state~1'
    y = 'stress~1'
    z = 'strain~1'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'isoharden elasticity g
              mandel_stress vonmises
              yield yield_zero normality eprate Eprate Erate Eerate
              ri_flowrate rd_flowrate flowrate integrate_ep integrate_stress effective_strain_rate
              mixed mixed_old'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'mixed_state equivalent_plastic_strain gamma_rate_ri'
    residuals = 'stress_residual equivalent_plastic_strain_residual complementarity'
  []
[]
