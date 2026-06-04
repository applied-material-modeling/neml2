# neml2
[Tensors]
  [end_time]
    type = Python
    expr = 'Scalar(torch.logspace(1.0, 4.0, 20, dtype=torch.float64))'
  []
  [times]
    type = Python
    expr = 'Scalar(end_time.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1))'
  []
  [max_stress]
    type = Python
    expr = 'SR2.fill(120.0, 0.0, 0.0, 0.0, 0.0, 0.0).dynamic_batch.expand(20)'
  []
  [stresses]
    type = Python
    expr = 'SR2(max_stress.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).reshape(100, 1, 1))'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'S'
    force_SR2_values = 'stresses'
    save_as = 'result.pt'
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
    cauchy_stress = 'S'
    mandel_stress = 'M'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'M'
    invariant = 'effective_stress'
  []
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 500
  []
  [yield_surface]
    type = YieldFunction
    yield_stress = 60
    effective_stress = 'effective_stress'
    isotropic_hardening = 'isotropic_hardening'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 2
  []
  [flow]
    type = ComposedModel
    models = 'vonmises yield_surface'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'yield_function'
    from = 'isotropic_hardening M'
    to = 'isotropic_hardening_direction flow_direction'
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
  []
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'plastic_strain'
  []
  [surface]
    type = ComposedModel
    models = 'isoharden yield_surface flow_rate normality eprate Eprate integrate_ep integrate_Ep'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'surface'
    unknowns = 'plastic_strain equivalent_plastic_strain'
    residuals = 'plastic_strain_residual equivalent_plastic_strain_residual'
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
    unknowns_SR2 = 'plastic_strain'
    unknowns_Scalar = 'equivalent_plastic_strain'
  []
  [return_map]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [stress_update]
    type = ComposedModel
    models = 'mandel_stress vonmises return_map'
  []
  [elastic_strain]
    type = LinearIsotropicElasticity
    coefficients = '3e4 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    compliance = true
    stress = 'S'
    strain = 'elastic_strain'
  []
  [total_strain]
    type = SR2LinearCombination
    to = 'total_strain'
    from = 'elastic_strain plastic_strain'
    weights = '1 1'
  []
  [model]
    type = ComposedModel
    models = 'stress_update elastic_strain total_strain'
    additional_outputs = 'equivalent_plastic_strain plastic_strain'
  []
[]
