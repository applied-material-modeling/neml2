[Tensors]
  [end_time]
    type = LogspaceScalar
    start = 0
    end = 1
    nstep = 20
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = end_time
    nstep = 100
  []
  [start_temperature]
    type = LinspaceScalar
    start = 300
    end = 500
    nstep = 20
  []
  [end_temperature]
    type = LinspaceScalar
    start = 1800
    end = 1200
    nstep = 20
  []
  [temperatures]
    type = LinspaceScalar
    start = start_temperature
    end = end_temperature
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
  #####################################################################################
  # Compute the invariant plastic flow direction and scalar trial stress
  #####################################################################################
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
  [trial_flow_direction]
    type = AssociativeJ2FlowDirection
    mandel_stress = 'S_trial'
    flow_direction = 'N_trial'
  []
  [vonmises_trial]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'S_trial'
    invariant = 's_trial'
  []
  [trial_state]
    type = ComposedModel
    models = 'trial_elastic_strain trial_cauchy_stress trial_flow_direction vonmises_trial'
  []

  #####################################################################################
  # Scalar-level stress update for the implicit solve
  #####################################################################################
  [trial_stress_update]
    type = LinearIsotropicElasticJ2TrialStressUpdate
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    elastic_trial_stress = 's_trial'
    equivalent_plastic_strain = 'equivalent_plastic_strain'
    updated_trial_stress = 'vonmises_stress'
  []
  [rom]
    type = TorchScriptFlowRate
    von_mises_stress = 'vonmises_stress'
    temperature = 'temperature'
    equivalent_plastic_strain_rate = 'equivalent_plastic_strain_rate'
    torch_script = 'gold/surrogate.pt'
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
  []
  [rate]
    type = ComposedModel
    models = 'trial_stress_update rom integrate_ep'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'rate'
    unknowns = 'equivalent_plastic_strain'
    residuals = 'equivalent_plastic_strain_residual'
  []
[]

[Solvers]
  [newton]
    type = Newton
    abs_tol = 1e-8
    rel_tol = 1e-6
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [predictor]
    type = LinearExtrapolationPredictor
    unknowns_Scalar = 'equivalent_plastic_strain'
  []
  [radial_return]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []

  #####################################################################################
  # Full tensor plastic strain update after the scalar solve
  #####################################################################################
  [ep_rate]
    type = ScalarVariableRate
    variable = 'equivalent_plastic_strain'
    time = 't'
  []
  [plastic_strain_rate_model]
    type = AssociativePlasticFlow
    flow_direction = 'N_trial'
    flow_rate = 'equivalent_plastic_strain_rate'
    plastic_strain_rate = 'plastic_strain_rate'
  []
  [plastic_strain_update]
    type = SR2ForwardEulerTimeIntegration
    variable = 'plastic_strain'
  []
  [plastic_update]
    type = ComposedModel
    models = 'ep_rate plastic_strain_rate_model plastic_strain_update'
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
  [stress_update]
    type = ComposedModel
    models = 'elastic_strain cauchy_stress'
  []
  [model]
    type = ComposedModel
    models = 'trial_state radial_return plastic_update stress_update'
    additional_outputs = 'equivalent_plastic_strain plastic_strain'
  []
[]
