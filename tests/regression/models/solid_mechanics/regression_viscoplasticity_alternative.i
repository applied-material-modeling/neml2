[Solvers]
  [newton]
    type = NewtonNonlinearSolver
  []
[]

[Models]
  [Ee]
    type = ElasticStrain
  []
  [S]
    type = CauchyStressFromElasticStrain
    E = 1e5
    nu = 0.3
  []
  [M]
    type = IsotropicMandelStress
  []
  [gamma]
    type = LinearIsotropicHardening
    K = 1000
  []
  [j2]
    type = J2StressMeasure
  []
  [f]
    type = IsotropicHardeningYieldFunction
    stress_measure = j2
    yield_stress = 5
  []
  [gammarate]
    type = PerzynaPlasticFlowRate
    eta = 100
    n = 2
  []
  [Np]
    type = AssociativePlasticFlowDirection
    yield_function = f
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
    yield_function = f
  []
  [Eprate]
    type = PlasticStrainRate
  []
  [rate]
    type = ComposedModel
    models = 'Ee S M gamma f gammarate Np eprate Eprate'
  []
  [surface]
    type = ImplicitTimeIntegration
    rate = rate
  []
  [return_map]
    type = ImplicitUpdate
    implicit_model = surface
    solver = newton
    additional_outputs = 'state plastic_strain; state internal_state equivalent_plastic_strain'
  []
  [model]
    type = ComposedModel
    models = 'return_map Ee S'
  []
[]
