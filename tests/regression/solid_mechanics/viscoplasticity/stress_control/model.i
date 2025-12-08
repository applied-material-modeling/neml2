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
  [szz]
    type = FullScalar
    batch_shape = '(20)'
    value = 0
  []
[]

[Drivers]
  [driver]
    type = SDTSolidMechanicsDriver
    model = 'model'
    prescribed_time = 'times'
    control = 'STRESS'
    prescribed_stress = 'stresses'
    predictor = LINEAR_EXTRAPOLATION
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
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
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'forces/S'
    mandel_stress = 'forces/M'
  []
  [vonmises]
    type = SR2Invariant
    tensor = 'forces/M'
    invariant = 'forces/s'
    invariant_type = 'VONMISES'
  []
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 500
  []
  [yield]
    type = YieldFunction
    yield_stress = 60
    effective_stress = 'forces/s'
    isotropic_hardening = 'state/internal/k'
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
    function = 'state/internal/fp'
    from = 'state/internal/k forces/M'
    to = 'state/internal/Nk state/internal/NM'
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/internal/ep'
  []
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/internal/Ep'
  []
  [surface]
    type = ComposedModel
    models = "isoharden yield flow_rate normality
              eprate Eprate
              integrate_ep integrate_Ep"
  []
  [return_map]
    type = ImplicitUpdate
    implicit_model = 'surface'
    solver = 'newton'
  []
  [stress_update]
    type = ComposedModel
    models = "mandel_stress vonmises return_map"
  []
  [elastic_strain]
    type = LinearIsotropicElasticity
    coefficients = '3e4 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    compliance = true
    stress = 'forces/S'
  []
  [total_strain]
    type = SR2LinearCombination
    to_var = 'state/E'
    from_var = 'state/internal/Ee state/internal/Ep'
  []
  [model]
    type = ComposedModel
    models = 'stress_update elastic_strain total_strain'
    additional_outputs = 'state/internal/ep state/internal/Ep'
  []
[]
