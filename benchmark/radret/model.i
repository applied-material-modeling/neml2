# neml2
[Settings]
  example_batch_shape = '(${nbatch},)'
[]

[Tensors]
  # HIT-substituted shim so the verbatim triple-quoted Python blocks below
  # can reference nbatch as a bare identifier. ${...} substitution only
  # works inside single-line single-quoted HIT strings; triple-quoted
  # blocks are passed verbatim to the Python eval namespace.
  [nbatch]
    type = Python
    expr = '${nbatch}'
  []
  # end_time = LogspaceScalar(-1, 5, nbatch) -> shape (nbatch,)
  [end_time]
    type = Python
    expr = '''
      Scalar(torch.logspace(-1.0, 5.0, nbatch, dtype=torch.float64))
    '''
  []
  # times = LinspaceScalar(0, end_time, 100) -> shape (100, nbatch)
  [times]
    type = Python
    expr = '''
      Scalar(
          end_time.data.unsqueeze(0)
          * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1)
      )
    '''
  []
  # max_strain = FillSR2(exx=0.1, eyy=-0.05, ezz=-0.05) broadcast to (nbatch, 6)
  [max_strain]
    type = Python
    expr = '''
      SR2.fill(0.1, -0.05, -0.05).dynamic_batch.expand(nbatch)
    '''
  []
  # strains = LinspaceSR2(0, max_strain, 100) -> shape (100, nbatch, 6)
  [strains]
    type = Python
    expr = '''
      SR2(
          max_strain.data.unsqueeze(0)
          * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).reshape(100, 1, 1)
      )
    '''
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'strain'
    force_SR2_values = 'strains'
  []
[]

[Models]
  ###############################################################################
  # Use the trial state to precalculate invariant flow directions
  # prior to radial return
  ###############################################################################
  [trial_elastic_strain]
    type = SR2LinearCombination
    from = 'strain plastic_strain~1'
    to = 'elastic_strain'
    weights = '1 -1'
  []
  # The elastic response (cauchy_stress + mandel_stress) is reused from the
  # single instances defined below for the actual return map. Each ComposedModel
  # scope binds 'elastic_strain' to its own value (trial here, current in
  # 'surface'), so the same elasticity instance -- and its parameters -- serves
  # both, rather than duplicating the model.
  [trial_isoharden]
    type = LinearIsotropicHardening
    equivalent_plastic_strain = 'equivalent_plastic_strain~1'
    isotropic_hardening = 'trial_isoharden'
    hardening_modulus = 1000
  []
  [trial_kinharden]
    type = LinearKinematicHardening
    kinematic_plastic_strain = 'kinematic_plastic_strain~1'
    back_stress = 'trial_backstress'
    hardening_modulus = 1000
  []
  [trial_overstress]
    type = SR2LinearCombination
    from = 'mandel_stress trial_backstress'
    to = 'trial_overstress'
    weights = '1 -1'
  []
  [trial_vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'trial_overstress'
    invariant = 'trial_effective_stress'
  []
  [trial_yield]
    type = YieldFunction
    yield_stress = 5
    yield_function = 'trial_yield_function'
    effective_stress = 'trial_effective_stress'
    isotropic_hardening = 'trial_isoharden'
  []
  [trial_flow]
    type = ComposedModel
    models = 'trial_overstress trial_vonmises trial_yield'
  []
  [trial_normality]
    type = Normality
    model = 'trial_flow'
    function = 'trial_yield_function'
    from = 'mandel_stress trial_isoharden trial_backstress'
    to = 'NM Nk NX'
  []
  [trial_state]
    type = ComposedModel
    models = 'trial_elastic_strain cauchy_stress mandel_stress
              trial_isoharden trial_kinharden trial_normality'
  []
  ###############################################################################
  # The actual radial return:
  # Since the flow directions are invariant, we only need to integrate
  # the consistency parameter.
  ###############################################################################
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 1000
  []
  [kinharden]
    type = LinearKinematicHardening
    hardening_modulus = 1000
  []
  [trial_flow_rate]
    type = ScalarVariableRate
    variable = 'gamma'
  []
  [plastic_strain_rate]
    type = AssociativePlasticFlow
    flow_direction = 'NM'
    flow_rate = 'gamma_rate'
  []
  [plastic_strain]
    type = SR2ForwardEulerTimeIntegration
    variable = 'plastic_strain'
  []
  [elastic_strain]
    type = SR2LinearCombination
    from = 'strain plastic_strain'
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
  [yield_surface]
    type = YieldFunction
    yield_stress = 5
    isotropic_hardening = 'isotropic_hardening'
  []
  [equivalent_plastic_strain_rate]
    type = AssociativeIsotropicPlasticHardening
    isotropic_hardening_direction = 'Nk'
    flow_rate = 'gamma_rate'
  []
  [equivalent_plastic_strain]
    type = ScalarForwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
  []
  [kinematic_plastic_strain_rate]
    type = AssociativeKinematicPlasticHardening
    kinematic_hardening_direction = 'NX'
    flow_rate = 'gamma_rate'
  []
  [kinematic_plastic_strain]
    type = SR2ForwardEulerTimeIntegration
    variable = 'kinematic_plastic_strain'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 2
  []
  [consistency]
    type = ScalarLinearCombination
    from = 'gamma_rate flow_rate'
    to = 'gamma_residual'
    weights = '1 -1'
  []
  [surface]
    type = ComposedModel
    models = 'trial_flow_rate
              plastic_strain_rate plastic_strain elastic_strain cauchy_stress mandel_stress
              kinematic_plastic_strain_rate kinematic_plastic_strain kinharden
              equivalent_plastic_strain_rate equivalent_plastic_strain isoharden
              overstress vonmises yield_surface
              flow_rate consistency'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'surface'
    unknowns = 'gamma'
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
    type = LinearExtrapolationPredictor
    unknowns_Scalar = 'gamma'
  []
  [return_map]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model0]
    type = ComposedModel
    models = 'trial_state return_map trial_flow_rate
              plastic_strain_rate plastic_strain
              equivalent_plastic_strain_rate equivalent_plastic_strain
              kinematic_plastic_strain_rate kinematic_plastic_strain'
    additional_outputs = 'gamma'
  []
  [model]
    type = ComposedModel
    models = 'model0 elastic_strain cauchy_stress'
    additional_outputs = 'plastic_strain equivalent_plastic_strain kinematic_plastic_strain'
  []
[]
