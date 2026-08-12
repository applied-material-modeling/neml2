# neml2
[Tensors]
  [end_time]
    type = Python
    expr = 'Scalar(torch.logspace(-1.0, 5.0, 20, dtype=torch.float64))'
  []
  [times]
    type = Python
    expr = 'Scalar(end_time.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1))'
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
    cauchy_stress = 'stress'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'mandel_stress'
    invariant = 'effective_stress'
  []
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 1000
  []
  [yield_surface]
    type = YieldFunction
    yield_stress = 5
    isotropic_hardening = 'isotropic_hardening'
  []
  [flow]
    type = ComposedModel
    models = 'vonmises yield_surface'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'yield_function'
    from = 'mandel_stress isotropic_hardening'
    to = 'flow_direction isotropic_hardening_direction'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 2
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
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
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'mandel_stress vonmises isoharden yield_surface normality flow_rate Eprate eprate
              Erate Eerate elasticity integrate_stress integrate_ep'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'stress equivalent_plastic_strain'
    residuals = 'stress_residual equivalent_plastic_strain_residual'
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
  # --- trial state: renamed twins (same leaves, different values) ---
  [pt_stress_rate]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    rate_form = true
    strain = 'E'
    stress = 'stress_trial'
  []
  [pt_stress]
    type = SR2ForwardEulerTimeIntegration
    variable = 'stress_trial'
    old_variable = 'stress~1'
  []
  [pt_mandel]
    type = IsotropicMandelStress
    cauchy_stress = 'stress_trial'
    mandel_stress = 'mandel_stress_trial'
  []
  [pt_vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'mandel_stress_trial'
    invariant = 'effective_stress_trial'
  []
  [pt_isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 1000
    equivalent_plastic_strain = 'equivalent_plastic_strain~1'
    isotropic_hardening = 'isotropic_hardening_trial'
  []
  [pt_yield]
    type = YieldFunction
    yield_stress = 5
    effective_stress = 'effective_stress_trial'
    isotropic_hardening = 'isotropic_hardening_trial'
    yield_function = 'yield_function_trial'
  []
  [pt_flow]
    type = ComposedModel
    models = 'pt_vonmises pt_yield'
  []
  [pt_normality]
    type = Normality
    model = 'pt_flow'
    function = 'yield_function_trial'
    from = 'mandel_stress_trial isotropic_hardening_trial'
    to = 'flow_direction isotropic_hardening_direction'
  []

  # --- return path: residual blocks reused, only the integrators are new ---
  [pp_stress]
    type = SR2ForwardEulerTimeIntegration
    variable = 'stress'
    old_variable = 'stress~1'
  []
  [pp_ep]
    type = ScalarForwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
    old_variable = 'equivalent_plastic_strain~1'
  []
  [pred_path]
    type = ComposedModel
    models = 'Erate Eprate Eerate elasticity pp_stress eprate pp_ep mandel_stress vonmises
              isoharden yield_surface'
  []
  [pred_coupling]
    type = RateCondensation
    model = 'pred_path'
    rate = 'flow_rate'
    driving_force = 'yield_function'
    coupling = 'flow_coupling'
    trial_driving_force = 'trial_yield_function'
  []
  [pred_cd]
    type = CoordinateDescentPredictor
    rate_law = 'flow_rate'
    driving_force_input = 'yield_function'
    coupling = 'flow_coupling'
    trial_driving_force = 'trial_yield_function'
    rate = 'flow_rate'
    sweeps = 1
  []
  [ps_stress]
    type = SR2ForwardEulerTimeIntegration
    variable = 'stress_cd'
    old_variable = 'stress~1'
    rate = 'stress_rate'
  []
  [ps_ep]
    type = ScalarForwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain_cd'
    old_variable = 'equivalent_plastic_strain~1'
    rate = 'equivalent_plastic_strain_rate'
  []
  [pred_seed]
    type = ConstantExtrapolationPredictor
    unknowns_SR2 = 'stress'
    unknowns_Scalar = 'equivalent_plastic_strain'
    cold = 'stress:stress_cd equivalent_plastic_strain:equivalent_plastic_strain_cd'
  []
  [predictor]
    type = ComposedModel
    models = 'pt_stress_rate pt_stress pt_mandel pt_vonmises pt_isoharden pt_yield pt_normality
              Erate Eprate Eerate elasticity eprate pred_coupling pred_cd ps_stress ps_ep
              pred_seed'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
[]
