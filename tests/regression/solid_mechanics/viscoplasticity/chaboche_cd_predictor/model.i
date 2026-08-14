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
  [isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 100
    saturation_rate = 1.2
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
  [flow]
    type = ComposedModel
    models = 'overstress vonmises yield_surface'
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
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [X1rate]
    type = ChabochePlasticHardening
    back_stress = 'X1'
    C = 10000
    g = 100
    A = 1e-8
    a = 1.2
  []
  [X2rate]
    type = ChabochePlasticHardening
    back_stress = 'X2'
    C = 1000
    g = 9
    A = 1e-10
    a = 3.2
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
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
  []
  [integrate_X1]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X1'
  []
  [integrate_X2]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X2'
  []
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'stress'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'isoharden kinharden mandel_stress overstress vonmises yield_surface normality
              flow_rate eprate Eprate X1rate X2rate Erate Eerate elasticity integrate_stress
              integrate_ep integrate_X1 integrate_X2'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'stress equivalent_plastic_strain X1 X2'
    residuals = 'stress_residual equivalent_plastic_strain_residual X1_residual X2_residual'
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
  # ================= trial state: renamed twins of residual leaves =============
  # Same physics as the residual chain, evaluated at the elastic trial state and
  # the OLD internal variables. Different values under the same roles, so these
  # are the blocks that genuinely have to be duplicated.
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
  [pt_kinharden]
    type = SR2LinearCombination
    from = 'X1~1 X2~1'
    to = 'back_stress_trial'
    weights = '1 1'
  []
  [pt_overstress]
    type = SR2LinearCombination
    to = 'overstress_trial'
    from = 'mandel_stress_trial back_stress_trial'
    weights = '1 -1'
  []
  [pt_vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'overstress_trial'
    invariant = 'effective_stress_trial'
  []
  [pt_isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 100
    saturation_rate = 1.2
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
    models = 'pt_overstress pt_vonmises pt_yield'
  []
  # The frozen directions are emitted under their PLAIN names, which is what lets
  # the residual's own rate leaves be reused below without a rename.
  [pt_normality]
    type = Normality
    model = 'pt_flow'
    function = 'yield_function_trial'
    from = 'mandel_stress_trial isotropic_hardening_trial'
    to = 'flow_direction isotropic_hardening_direction'
  []

  # ================= the return path ==========================================
  # Explicit Euler, so each back-stress rate is evaluated at X~1 rather than at
  # the iterate. Everything else -- Eprate, Erate, Eerate, elasticity, eprate,
  # kinharden, mandel_stress, overstress, vonmises, isoharden, yield_surface --
  # is the residual's block, listed as-is.
  [pp_X1rate]
    type = ChabochePlasticHardening
    back_stress = 'X1~1'
    back_stress_rate = 'X1_rate'
    C = 10000
    g = 100
    A = 1e-8
    a = 1.2
  []
  [pp_X2rate]
    type = ChabochePlasticHardening
    back_stress = 'X2~1'
    back_stress_rate = 'X2_rate'
    C = 1000
    g = 9
    A = 1e-10
    a = 3.2
  []
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
  [pp_X1]
    type = SR2ForwardEulerTimeIntegration
    variable = 'X1'
    old_variable = 'X1~1'
  []
  [pp_X2]
    type = SR2ForwardEulerTimeIntegration
    variable = 'X2'
    old_variable = 'X2~1'
  []
  [pred_path]
    type = ComposedModel
    models = 'Erate Eprate Eerate elasticity pp_stress eprate pp_ep pp_X1rate pp_X1 pp_X2rate
              pp_X2 kinharden mandel_stress overstress vonmises isoharden yield_surface'
  []

  # ================= condensation + coordinate descent ========================
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

  # ================= seed: the same rates, integrated to _cd names ============
  # A second set of integrators only because the cold gate needs the seed under
  # names distinct from the unknowns. The rates they consume are the reused
  # residual blocks' outputs.
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
  [ps_X1]
    type = SR2ForwardEulerTimeIntegration
    variable = 'X1_cd'
    old_variable = 'X1~1'
    rate = 'X1_rate'
  []
  [ps_X2]
    type = SR2ForwardEulerTimeIntegration
    variable = 'X2_cd'
    old_variable = 'X2~1'
    rate = 'X2_rate'
  []
  [pred_seed]
    type = ConstantExtrapolationPredictor
    unknowns_SR2 = 'stress X1 X2'
    unknowns_Scalar = 'equivalent_plastic_strain'
    cold = 'stress:stress_cd X1:X1_cd X2:X2_cd equivalent_plastic_strain:equivalent_plastic_strain_cd'
  []
  [predictor]
    type = ComposedModel
    models = 'pt_stress_rate pt_stress pt_mandel pt_kinharden pt_overstress pt_vonmises
              pt_isoharden pt_yield pt_normality Erate Eprate Eerate elasticity eprate pp_X1rate
              pp_X2rate pred_coupling pred_cd ps_stress ps_ep ps_X1 ps_X2 pred_seed'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
[]
