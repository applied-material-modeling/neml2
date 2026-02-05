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
    type = SDTSolidMechanicsDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_strain = 'strains'
    prescribed_temperature = 'temperatures'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Models]
  [isoharden]
    type = SlopeSaturationVoceIsotropicHardening
    saturated_hardening = 100
    initial_hardening_rate = 1200.0
    isotropic_hardening_rate = 'state/internal/k_rate_base'
  []
  [isoharden_recovery]
    type = PowerLawIsotropicHardeningStaticRecovery
    n = 2.0
    tau = 1000.0
  []
  [isoharden_total]
    type = ScalarLinearCombination
    to_var = 'state/internal/k_rate_before_anneal'
    from_var = 'state/internal/k_rate_base state/internal/k_recovery_rate'
    coefficients = '1 1'
  []
  [anneal_isoharden]
    type = ScalarTwoStageThermalAnnealing
    base_rate = 'state/internal/k_rate_before_anneal'
    base = 'state/internal/k'
    modified_rate = 'state/internal/k_rate'
    temperature = 'forces/T'
    T1 = 1000.0
    T2 = 1200.0
    tau = 0.01
  []
  [kinharden]
    type = SR2LinearCombination
    from_var = 'state/internal/X1 state/internal/X2'
    to_var = 'state/internal/X'
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
    yield_stress = 5
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
    from = 'state/internal/M'
    to = 'state/internal/NM'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 2
  []
  [X1rate]
    type = FredrickArmstrongPlasticHardening
    back_stress = 'state/internal/X1'
    back_stress_rate = 'state/internal/X1_rate_base'
    C = 10000
    g = 100
  []
  [X1_recovery]
    type = PowerLawKinematicHardeningStaticRecovery
    back_stress = 'state/internal/X1'
    n = 2.0
    tau = 1000.0
  []
  [X1_total]
    type = SR2LinearCombination
    to_var = 'state/internal/X1_rate_before_anneal'
    from_var = 'state/internal/X1_rate_base state/internal/X1_recovery_rate'
    coefficients = '1 1'
  []
  [anneal_X1]
    type = SR2TwoStageThermalAnnealing
    base_rate = 'state/internal/X1_rate_before_anneal'
    base = 'state/internal/X1'
    modified_rate = 'state/internal/X1_rate'
    temperature = 'forces/T'
    T1 = 1000.0
    T2 = 1200.0
    tau = 0.01
  []
  [X2rate]
    type = FredrickArmstrongPlasticHardening
    back_stress = 'state/internal/X2'
    back_stress_rate = 'state/internal/X2_rate_base'
    C = 1000
    g = 9
  []
  [X2_recovery]
    type = PowerLawKinematicHardeningStaticRecovery
    back_stress = 'state/internal/X2'
    n = 2.5
    tau = 500.0
  []
  [X2_total]
    type = SR2LinearCombination
    to_var = 'state/internal/X2_rate_before_anneal'
    from_var = 'state/internal/X2_rate_base state/internal/X2_recovery_rate'
    coefficients = '1 1'
  []
  [anneal_X2]
    type = SR2TwoStageThermalAnnealing
    base_rate = 'state/internal/X2_rate_before_anneal'
    base = 'state/internal/X2'
    modified_rate = 'state/internal/X2_rate'
    temperature = 'forces/T'
    T1 = 1000.0
    T2 = 1200.0
    tau = 0.01
  []
  [Eprate]
    type = AssociativePlasticFlow
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
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    rate_form = true
  []
  [integrate_k]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/internal/k'
  []
  [integrate_X1]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/internal/X1'
  []
  [integrate_X2]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/internal/X2'
  []
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/S'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'isoharden isoharden_recovery isoharden_total anneal_isoharden kinharden mandel_stress overstress vonmises yield normality flow_rate Eprate X1rate X1_recovery X1_total anneal_X1 X2rate X2_recovery X2_total anneal_X2 Erate Eerate elasticity integrate_stress integrate_k integrate_X1 integrate_X2'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
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
