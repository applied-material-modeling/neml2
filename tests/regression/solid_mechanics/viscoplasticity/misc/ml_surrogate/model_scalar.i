# neml2
[Tensors]
  [end_time]
    type = Python
    expr = 'Scalar(torch.logspace(0.0, 1.0, 20, dtype=torch.float64))'
  []
  [times]
    type = Python
    expr = 'Scalar(end_time.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1))'
  []
  [start_temperature]
    type = Python
    expr = 'Scalar.linspace(300.0, 500.0, 20)'
  []
  [end_temperature]
    type = Python
    expr = 'Scalar.linspace(1800.0, 1200.0, 20)'
  []
  [temperatures]
    type = Python
    expr = 'Scalar(start_temperature.data.unsqueeze(0) + (end_temperature.data - start_temperature.data).unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1))'
  []
  [max_strain]
    type = Python
    expr = 'SR2.fill(0.1, -0.05, -0.05, 0.0, 0.0, 0.0).dynamic_batch.expand(20)'
  []
  [strains]
    type = Python
    expr = 'SR2(max_strain.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).reshape(100, 1, 1))'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_SR2_names = 'E'
    prescribed_SR2_values = 'strains'
    prescribed_Scalar_names = 'temperature'
    prescribed_Scalar_values = 'temperatures'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
    atol = 1e-5
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
    type = SurrogateFlowRate
    von_mises_stress = 'vonmises_stress'
    temperature = 'temperature'
    equivalent_plastic_strain_rate = 'equivalent_plastic_strain_rate'
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
