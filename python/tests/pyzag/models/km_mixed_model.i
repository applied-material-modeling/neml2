[Tensors]
  [T_controls]
    type = Scalar
    values = '300 347.36842105 394.73684211 442.10526316 489.47368421 536.84210526 584.21052632 631.57894737 678.94736842 726.31578947 773.68421053 821.05263158 868.42105263 915.78947368 963.15789474 1010.52631579 1057.89473684 1105.26315789 1152.63157895 1200'
    batch_shape = '(20)'
    intermediate_dimension = 1
  []
  [mu_values]
    type = Scalar
    values = '76670.48346056 75465.18012589 74314.80514263 73374.72880675 72651.54680595 71928.36480514 71120.75130575 70035.97830454 68951.20530333 67842.26597027 66399.97991161 65315.20691041 63884.85335476 62763.98151868 61373.80474086 59927.44073925 58481.07673765 56544.43551627 54599.93973483 52791.98473282'
    batch_shape = '(20)'
    intermediate_dimension = 1
  []
  [E_test]
    type = Scalar
    values = '1.0e5'
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
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/M'
    invariant = 'state/internal/s'
  []
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 1000
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
    C = 'C'
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
    yield_function = 'state/internal/fp_alt'
    flow_rate = 'state/internal/gamma_rate_rd'
  []
  [effective_strain_rate]
    type = SR2Invariant
    invariant_type = 'EFFECTIVE_STRAIN'
    tensor = 'state/E_rate'
    invariant = 'forces/effective_strain_rate'
  []
  [g]
    type = KocksMeckingActivationEnergy
    shear_modulus = 'mu'
    k = 1.38064e-20
    b = 2.474e-7
    eps0 = 1e10
  []
  [flowrate]
    type = KocksMeckingFlowSwitch
    g0 = 'g0'
    rate_independent_flow_rate = 'state/internal/gamma_rate_ri'
    rate_dependent_flow_rate = 'state/internal/gamma_rate_rd'
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
    coefficients = 'E_test 0.3'
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
  [mixed]
    type = MixedControlSetup
    above_variable = 'state/S'
    below_variable = 'state/E'
  []
  [mixed_old]
    type = MixedControlSetup
    control = 'old_forces/control'
    mixed_state = 'old_state/mixed_state'
    fixed_values = 'old_forces/fixed_values'
    above_variable = 'old_state/S'
    below_variable = 'old_state/E'
  []
  [rename]
    type = CopySR2
    from = 'residual/S'
    to = 'residual/mixed_state'
  []
  [implicit_rate]
    type = ComposedModel
    models = "isoharden elasticity g
              mandel_stress vonmises
              yield yield_zero normality eprate Eprate Erate Eerate
              ri_flowrate rd_flowrate flowrate integrate_ep integrate_stress effective_strain_rate
              mixed mixed_old rename"
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
  []
[]
