[Tensors]
  [end_time]
    type = LogspaceScalar
    start = 1
    end = 4
    nstep = 20
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = end_time
    nstep = 100
  []
  [sxx]
    type = FullScalar
    batch_shape = '(20)'
    value = 120
  []
  [syy]
    type = FullScalar
    batch_shape = '(20)'
    value = 0
  []
  [szz]
    type = FullScalar
    batch_shape = '(20)'
    value = 0
  []
  [max_stress]
    type = FillSR2
    values = 'sxx syy szz'
  []
  [stresses]
    type = LinspaceSR2
    start = 0
    end = max_stress
    nstep = 100
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'S'
    force_SR2_values = 'stresses'
    save_as = 'result.pt'
    predictor = 'LINEAR_EXTRAPOLATION'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Models]
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'S'
    mandel_stress = 'M'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'M'
    invariant = 'effective_stress'
  []
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 500
  []
  [yield]
    type = YieldFunction
    yield_stress = 60
    effective_stress = 'effective_stress'
    isotropic_hardening = 'isotropic_hardening'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 2
  []
  [flow]
    type = ComposedModel
    models = 'vonmises yield'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'yield_function'
    from = 'isotropic_hardening M'
    to = 'isotropic_hardening_direction flow_direction'
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
  []
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'plastic_strain'
  []
  [surface]
    type = ComposedModel
    models = 'isoharden yield flow_rate normality eprate Eprate integrate_ep integrate_Ep'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'surface'
    unknowns = 'plastic_strain equivalent_plastic_strain'
    residuals = 'plastic_strain_residual equivalent_plastic_strain_residual'
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
  [return_map]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
  [stress_update]
    type = ComposedModel
    models = 'mandel_stress vonmises return_map'
  []
  [elastic_strain]
    type = LinearIsotropicElasticity
    coefficients = '3e4 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    compliance = true
    stress = 'S'
    strain = 'elastic_strain'
  []
  [total_strain]
    type = SR2LinearCombination
    to = 'total_strain'
    from = 'elastic_strain plastic_strain'
    weights = '1 1'
  []
  [model]
    type = ComposedModel
    models = 'stress_update elastic_strain total_strain'
    additional_outputs = 'equivalent_plastic_strain plastic_strain'
  []
[]
