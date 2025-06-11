[Tensors]
  [T_hardening]
    type = Scalar
    values = '293.15 548.15 823.15 1023.15 1173.15 1273.15 1373.15 1673.15 5000'
    batch_shape = '(9)'
  []
  [R]
    type = Scalar
    values = '153.4 154.7 150.6 57.9 0.0 0.0 0.0 0.0 0.0'
    batch_shape = '(9)'
  []
  [X1_rate]
    type = Scalar
    values = '156435 100631 64341 56232 0.05 0.0 0.0 0.0 0.0'
    batch_shape = '(9)'
  []
  [X2_rate]
    type = Scalar
    values = '6134 5568 5227 4108 292 0.0 0.0 0.0 0.0'
    batch_shape = '(9)'
  []
  [yield_stress]
    type = Scalar
    values = '125.6 97.6 90.9 71.4 66.2 31.82 19.73 2.1 2.1'
    batch_shape = '(9)'
  []
  [T_youngs]
    type = Scalar
    values = '293.15 373.15 473.15 573.15 673.15 773.15 873.15 973.15 1073.15 1173.15 1273.15 1373.15 1473.15 1573.15 5000'
    batch_shape = '(15)'
  []
  [youngs_modulus]
    type = Scalar
    values = '195600 191200 185700 179600 172600 164500 155000 144100 131400 116800 100000 80000 57000 30000 30000'
    batch_shape = '(15)'
  []
  [times]
    type = ScalarVTestTimeSeries
    vtest = 'chaboche.vtest'
    variable = 'time'
  []
  [strains]
    type = SR2VTestTimeSeries
    vtest = 'chaboche.vtest'
    variable = 'strain'
  []
  [stresses]
    type = SR2VTestTimeSeries
    vtest = 'chaboche.vtest'
    variable = 'stress'
  []
  [temperatures]
    type = ScalarVTestTimeSeries
    vtest = 'chaboche.vtest'
    variable = 'temperature'
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
    variables = 'output.state/S'
    references = 'stresses'
    atol = 1e-4
    rtol = 1e-5
  []
[]

[Solvers]
  [newton]
    type = Newton
  []
[]

[Models]
  [saturated_hardening_func]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = 'T_hardening'
    ordinate = 'R'
  []
  [youngs_modulus_func]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = 'T_youngs'
    ordinate = 'youngs_modulus'
  []
  [yield_stress_func]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = 'T_hardening'
    ordinate = 'yield_stress'
  []
  [X1_rate_func]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = 'T_hardening'
    ordinate = 'X1_rate'
  []
  [X2_rate_func]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = 'T_hardening'
    ordinate = 'X2_rate'
  []
  [isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 'saturated_hardening_func'
    saturation_rate = 6.9
  []
  [kinharden]
    type = SR2LinearCombination
    from_var = 'state/internal/X1 state/internal/X2'
    to_var = 'state/internal/X'
  []
  [elastic_strain]
    type = SR2LinearCombination
    from_var = 'forces/E state/internal/Ep'
    to_var = 'state/internal/Ee'
    coefficients = '1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = 'youngs_modulus_func 0.29'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
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
    yield_stress = 'yield_stress_func'
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
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [X1rate]
    type = ChabochePlasticHardening
    back_stress = 'state/internal/X1'
    C = 'X1_rate_func'
    g = 1151.95
    A = 0
    a = 2
  []
  [X2rate]
    type = ChabochePlasticHardening
    back_stress = 'state/internal/X2'
    C = 'X2_rate_func'
    g = 38.53 
    A = 0
    a = 2
  []
  [Eprate]
    type = AssociativePlasticFlow
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
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/internal/Ep'
  []
  [consistency]
    type = RateIndependentPlasticFlowConstraint
  []
  [surface]
    type = ComposedModel
    models = 'isoharden kinharden elastic_strain elasticity
              mandel_stress overstress vonmises
              yield normality eprate X1rate X2rate Eprate
              consistency integrate_ep integrate_X1 integrate_X2 integrate_Ep'
  []
  [return_map]
    type = ImplicitUpdate
    implicit_model = 'surface'
    solver = 'newton'
  []
  [model]
    type = ComposedModel
    models = 'return_map elastic_strain elasticity'
    additional_outputs = 'state/internal/Ep state/internal/X1 state/internal/X2 state/internal/ep'
  []
[]
