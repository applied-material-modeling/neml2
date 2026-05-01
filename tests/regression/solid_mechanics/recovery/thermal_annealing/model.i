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
    nstep = 100
  []
  [exx]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.1
  []
  [eyy]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.05
  []
  [ezz]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.05
  []
  [max_strain]
    type = FillSR2
    values = 'exx eyy ezz'
  []
  [strains]
    type = LinspaceSR2
    start = 0
    end = max_strain
    nstep = 100
  []
  [start_temperature]
    type = LinspaceScalar
    start = 900
    end = 900
    nstep = 20
  []
  [end_temperature]
    type = LinspaceScalar
    start = 1300
    end = 1300
    nstep = 20
  []
  [temperatures]
    type = LinspaceScalar
    start = start_temperature
    end = end_temperature
    nstep = 100
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
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Models]
  # Isotropic hardening: alias to avoid naming conflict between Voce and recovery
  [alias_k_voce]
    type = ScalarLinearCombination
    from = 'isotropic_hardening'
    to = 'k_voce'
    weights = '1'
  []
  [alias_k_recv]
    type = ScalarLinearCombination
    from = 'isotropic_hardening'
    to = 'k_recv'
    weights = '1'
  []
  [isoharden]
    type = SlopeSaturationVoceIsotropicHardening
    saturated_hardening = 100
    initial_hardening_rate = 1200.0
    isotropic_hardening = 'k_voce'
  []
  [isoharden_recovery]
    type = PowerLawIsotropicHardeningStaticRecovery
    n = 2.0
    tau = 1000.0
    isotropic_hardening = 'k_recv'
  []
  [isoharden_total]
    type = ScalarLinearCombination
    from = 'k_voce_rate k_recv_rate'
    to = 'k_rate_before_anneal'
    weights = '1 1'
  []
  [anneal_isoharden]
    type = ScalarTwoStageThermalAnnealing
    base_rate = 'k_rate_before_anneal'
    base = 'isotropic_hardening'
    temperature = 'temperature'
    modified_rate = 'isotropic_hardening_rate'
    T1 = 1000.0
    T2 = 1200.0
    tau = 0.01
  []

  # Kinematic hardening X1: alias to avoid output name conflict
  [alias_X1_fa]
    type = SR2LinearCombination
    from = 'X1'
    to = 'X1_fa'
    weights = '1'
  []
  [alias_X1_recv]
    type = SR2LinearCombination
    from = 'X1'
    to = 'X1_recv'
    weights = '1'
  []
  [X1rate]
    type = FredrickArmstrongPlasticHardening
    back_stress = 'X1_fa'
    flow_direction = 'flow_direction'
    C = 10000
    g = 100
  []
  [X1_recovery]
    type = PowerLawKinematicHardeningStaticRecovery
    back_stress = 'X1_recv'
    n = 2.0
    tau = 1000.0
  []
  [X1_total]
    type = SR2LinearCombination
    from = 'X1_fa_rate X1_recv_rate'
    to = 'X1_rate_before_anneal'
    weights = '1 1'
  []
  [anneal_X1]
    type = SR2TwoStageThermalAnnealing
    base_rate = 'X1_rate_before_anneal'
    base = 'X1'
    temperature = 'temperature'
    modified_rate = 'X1_rate'
    T1 = 1000.0
    T2 = 1200.0
    tau = 0.01
  []

  # Kinematic hardening X2: alias to avoid output name conflict
  [alias_X2_fa]
    type = SR2LinearCombination
    from = 'X2'
    to = 'X2_fa'
    weights = '1'
  []
  [alias_X2_recv]
    type = SR2LinearCombination
    from = 'X2'
    to = 'X2_recv'
    weights = '1'
  []
  [X2rate]
    type = FredrickArmstrongPlasticHardening
    back_stress = 'X2_fa'
    flow_direction = 'flow_direction'
    C = 1000
    g = 9
  []
  [X2_recovery]
    type = PowerLawKinematicHardeningStaticRecovery
    back_stress = 'X2_recv'
    n = 2.5
    tau = 500.0
  []
  [X2_total]
    type = SR2LinearCombination
    from = 'X2_fa_rate X2_recv_rate'
    to = 'X2_rate_before_anneal'
    weights = '1 1'
  []
  [anneal_X2]
    type = SR2TwoStageThermalAnnealing
    base_rate = 'X2_rate_before_anneal'
    base = 'X2'
    temperature = 'temperature'
    modified_rate = 'X2_rate'
    T1 = 1000.0
    T2 = 1200.0
    tau = 0.01
  []

  [kinharden]
    type = SR2LinearCombination
    from = 'X1 X2'
    to = 'back_stress'
    weights = '1 1'
  []
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'stress'
  []
  [overstress]
    type = SR2LinearCombination
    from = 'mandel_stress back_stress'
    to = 'overstress'
    weights = '1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'overstress'
    invariant = 'effective_stress'
  []
  [yield]
    type = YieldFunction
    yield_stress = 5
    isotropic_hardening = 'isotropic_hardening'
  []
  [flow]
    type = ComposedModel
    models = 'overstress vonmises yield'
    automatic_nonlinear_parameter = false
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'yield_function'
    from = 'mandel_stress'
    to = 'flow_direction'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 2
  []
  [Eprate]
    type = AssociativePlasticFlow
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
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    rate_form = true
  []
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'stress'
  []
  [integrate_k]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'isotropic_hardening'
  []
  [integrate_X1]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X1'
  []
  [integrate_X2]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X2'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'alias_k_voce alias_k_recv isoharden isoharden_recovery isoharden_total anneal_isoharden
              alias_X1_fa alias_X1_recv X1rate X1_recovery X1_total anneal_X1
              alias_X2_fa alias_X2_recv X2rate X2_recovery X2_total anneal_X2
              kinharden mandel_stress overstress vonmises yield
              normality flow_rate Eprate
              Erate Eerate elasticity
              integrate_stress integrate_k integrate_X1 integrate_X2'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'stress isotropic_hardening X1 X2'
    residuals = 'stress_residual isotropic_hardening_residual X1_residual X2_residual'
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
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_SR2 = 'stress X1 X2'
    unknowns_Scalar = 'isotropic_hardening'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
[]
