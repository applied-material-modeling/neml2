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
    expr = 'SR2(torch.tensor([0.1, -0.05, -0.05, 0.0, 0.0, 0.0], dtype=torch.float64).unsqueeze(0).expand(20, 6).contiguous())'
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
  [kinharden]
    type = LinearKinematicHardening
    hardening_modulus = 1000
    back_stress = 'X'
  []
  [elastic_strain]
    type = SR2LinearCombination
    from = 'E plastic_strain'
    to = 'elastic_strain'
    weights = '1 -1'
  []
  [elasticity]
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
    to = 'O'
    from = 'mandel_stress X'
    weights = '1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'O'
    invariant = 'effective_stress'
  []
  [yield]
    type = YieldFunction
    yield_stress = 1000
  []
  [flow]
    type = ComposedModel
    models = 'overstress vonmises yield'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'yield_function'
    from = 'mandel_stress X'
    to = 'flow_direction kinematic_hardening_direction'
  []
  [Kprate]
    type = AssociativeKinematicPlasticHardening
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [integrate_Kp]
    type = SR2BackwardEulerTimeIntegration
    variable = 'kinematic_plastic_strain'
  []
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'plastic_strain'
  []
  [consistency]
    type = FBComplementarity
    a = 'yield_function'
    a_inequality = 'LE'
    b = 'flow_rate'
  []
  [surface]
    type = ComposedModel
    models = 'kinharden elastic_strain elasticity
              mandel_stress overstress vonmises
              yield normality Kprate Eprate
              consistency integrate_Kp integrate_Ep'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'surface'
    unknowns = 'plastic_strain kinematic_plastic_strain flow_rate'
    residuals = 'plastic_strain_residual kinematic_plastic_strain_residual complementarity'
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
    unknowns_SR2 = 'plastic_strain kinematic_plastic_strain'
    unknowns_Scalar = 'flow_rate'
  []
  [return_map]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model]
    type = ComposedModel
    models = 'return_map elastic_strain elasticity'
    additional_outputs = 'plastic_strain kinematic_plastic_strain'
  []
[]
