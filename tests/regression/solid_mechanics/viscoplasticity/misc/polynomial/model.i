[Tensors]
  [end_time]
    type = LogspaceScalar
    start = -3
    end = -3
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
    start = 100
    end = 1000
    nstep = 20
  []
  [end_temperature]
    type = LinspaceScalar
    start = 200
    end = 1500
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
  [s1_0]
    type = FullScalar
    batch_shape = '(20)'
    value = 0
  []
  [s2_0]
    type = FullScalar
    batch_shape = '(20)'
    value = 0
  []
  [A0]
    type = Tensor
    values = "1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6
              1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6"
    base_shape = '(2,3,3)'
  []
  [A1]
    type = Tensor
    values = "1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6
              1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6
              1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6
              1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6
              1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6
              1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6"
    base_shape = '(2,3,3,4)'
  []
  [A2]
    type = Tensor
    values = "1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6
              1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6
              1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6
              1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6
              1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6
              1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6 1e-6 2e-6 3e-6 4e-6"
    base_shape = '(2,3,3,4)'
  []
  [s_lb]
    type = Tensor
    values = '0 50'
    base_shape = (2)
  []
  [s_ub]
    type = Tensor
    values = '50 100'
    base_shape = (2)
  []
  [T_lb]
    type = Tensor
    values = '0 300 600'
    base_shape = (3)
  []
  [T_ub]
    type = Tensor
    values = '300 600 1000'
    base_shape = (3)
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
    ic_Scalar_names = 's1 s2'
    ic_Scalar_values = 's1_0 s2_0'
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
  # Compute the invariant plastic flow direction since we are doing J2 radial return
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
  [trial_state]
    type = ComposedModel
    models = 'trial_elastic_strain trial_cauchy_stress trial_flow_direction'
  []

  #####################################################################################
  # Stress update (forward Euler in plastic strain using trial flow direction)
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

  #####################################################################################
  # ROM rate model and residuals
  #####################################################################################
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'stress'
    invariant = 'vonmises_stress'
  []
  [rom]
    type = TabulatedPolynomialModel
    jit = false
    von_mises_stress = 'vonmises_stress'
    temperature = 'temperature'
    internal_state_1 = 's1'
    internal_state_2 = 's2'
    equivalent_plastic_strain_rate = 'ep_rate_from_rom'
    internal_state_1_rate = 's1_rate'
    internal_state_2_rate = 's2_rate'
    A0 = 'A0'
    A1 = 'A1'
    A2 = 'A2'
    stress_tile_lower_bounds = 's_lb'
    stress_tile_upper_bounds = 's_ub'
    temperature_tile_lower_bounds = 'T_lb'
    temperature_tile_upper_bounds = 'T_ub'
  []
  [ep_residual]
    type = ScalarLinearCombination
    from = 'equivalent_plastic_strain_rate ep_rate_from_rom'
    to = 'equivalent_plastic_strain_residual'
    weights = '1 -1'
  []
  [integrate_s1]
    type = ScalarBackwardEulerTimeIntegration
    variable = 's1'
  []
  [integrate_s2]
    type = ScalarBackwardEulerTimeIntegration
    variable = 's2'
  []
  [rate]
    type = ComposedModel
    models = 'ep_rate plastic_strain_rate_model plastic_strain_update elastic_strain cauchy_stress vonmises rom ep_residual integrate_s1 integrate_s2'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'rate'
    unknowns = 'equivalent_plastic_strain s1 s2'
    residuals = 'equivalent_plastic_strain_residual s1_residual s2_residual'
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
    type = ConstantExtrapolationPredictor
    unknowns_Scalar = 'equivalent_plastic_strain s1 s2'
  []
  [radial_return]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model]
    type = ComposedModel
    models = 'trial_state radial_return plastic_update stress_update'
    additional_outputs = 'equivalent_plastic_strain plastic_strain s1 s2'
  []
[]
