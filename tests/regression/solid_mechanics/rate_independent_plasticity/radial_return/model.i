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
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'E'
    force_SR2_values = 'strains'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Models]
  ###############################################################################
  # Use the trial state to precalculate invariant flow directions
  # prior to radial return
  ###############################################################################
  [trial_elastic_strain]
    type = SR2LinearCombination
    from = 'E plastic_strain~1'
    to = 'Ee_trial'
    weights = '1 -1'
  []
  [trial_cauchy_stress]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'Ee_trial'
    stress = 'S_trial'
  []
  [trial_mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'S_trial'
    mandel_stress = 'M_trial'
  []
  [trial_isoharden]
    type = LinearIsotropicHardening
    equivalent_plastic_strain = 'equivalent_plastic_strain~1'
    isotropic_hardening = 'k_trial'
    hardening_modulus = 1000
  []
  [trial_kinharden]
    type = LinearKinematicHardening
    kinematic_plastic_strain = 'kinematic_plastic_strain~1'
    back_stress = 'X_trial'
    hardening_modulus = 1000
  []
  [trial_overstress]
    type = SR2LinearCombination
    to = 'O_trial'
    from = 'M_trial X_trial'
    weights = '1 -1'
  []
  [trial_vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'O_trial'
    invariant = 's_trial'
  []
  [trial_yield]
    type = YieldFunction
    yield_stress = 1000
    yield_function = 'fp_trial'
    effective_stress = 's_trial'
    isotropic_hardening = 'k_trial'
  []
  [trial_flow]
    type = ComposedModel
    models = 'trial_overstress trial_vonmises trial_yield'
  []
  [trial_normality]
    type = Normality
    model = 'trial_flow'
    function = 'fp_trial'
    from = 'M_trial k_trial X_trial'
    to = 'NM Nk NX'
  []
  [trial_state]
    type = ComposedModel
    models = "trial_elastic_strain trial_cauchy_stress trial_mandel_stress
              trial_isoharden trial_kinharden trial_normality"
  []
  ###############################################################################
  # The actual radial return:
  # Since the flow directions are invariant, we only need to solve
  # the consistency condition.
  ###############################################################################
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 1000
  []
  [kinharden]
    type = LinearKinematicHardening
    hardening_modulus = 1000
  []
  [plastic_strain_rate]
    type = AssociativePlasticFlow
    flow_direction = 'NM'
  []
  [plastic_strain]
    type = SR2ForwardEulerTimeIntegration
    variable = 'plastic_strain'
  []
  [elastic_strain]
    type = SR2LinearCombination
    from = 'E plastic_strain'
    to = 'elastic_strain'
    weights = '1 -1'
  []
  [cauchy_stress]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
  []
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'stress'
  []
  [overstress]
    type = SR2LinearCombination
    to = 'overstress'
    from = 'mandel_stress back_stress'
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
    yield_stress = 1000
    isotropic_hardening = 'isotropic_hardening'
  []
  [equivalent_plastic_strain_rate]
    type = AssociativeIsotropicPlasticHardening
    isotropic_hardening_direction = 'Nk'
  []
  [equivalent_plastic_strain]
    type = ScalarForwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
  []
  [kinematic_plastic_strain_rate]
    type = AssociativeKinematicPlasticHardening
    kinematic_hardening_direction = 'NX'
  []
  [kinematic_plastic_strain]
    type = SR2ForwardEulerTimeIntegration
    variable = 'kinematic_plastic_strain'
  []
  [consistency]
    type = FischerBurmeister
    a = 'yield_function'
    a_inequality = 'LE'
    b = 'flow_rate'
    jit = false
  []
  [surface]
    type = ComposedModel
    models = "plastic_strain_rate plastic_strain elastic_strain cauchy_stress mandel_stress
              kinematic_plastic_strain_rate kinematic_plastic_strain kinharden
              equivalent_plastic_strain_rate equivalent_plastic_strain isoharden
              overstress vonmises yield consistency"
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'surface'
    unknowns = 'flow_rate'
    residuals = 'complementarity'
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
  [model0]
    type = ComposedModel
    models = "trial_state return_map
              plastic_strain_rate plastic_strain
              equivalent_plastic_strain_rate equivalent_plastic_strain
              kinematic_plastic_strain_rate kinematic_plastic_strain"
  []
  [model]
    type = ComposedModel
    models = 'model0 elastic_strain cauchy_stress'
    additional_outputs = 'plastic_strain equivalent_plastic_strain kinematic_plastic_strain'
  []
[]
