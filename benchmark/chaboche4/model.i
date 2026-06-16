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
  # end_time = LogspaceScalar(5, 5, nbatch) -> shape (nbatch,)
  [end_time]
    type = Python
    expr = '''
      Scalar(torch.logspace(5.0, 5.0, nbatch, dtype=torch.float64))
    '''
  []
  # times = LinspaceScalar(0, end_time, 100) -> shape (100, nbatch)
  [times]
    type = Python
    expr = '''
      result = Scalar(
          end_time.data.unsqueeze(0)
          * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1)
      )
    '''
  []
  # max_strain = FillSR2(0.1, -0.05, -0.05) broadcast to (nbatch, 6)
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
  [isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 100
    saturation_rate = 1.2
  []
  [kinharden]
    type = SR2LinearCombination
    from = 'X1 X2 X3 X4'
    to = 'back_stress'
  []
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'stress'
  []
  [overstress]
    type = SR2LinearCombination
    from = 'mandel_stress back_stress'
    to = 'overstress'
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
  [X3rate]
    type = ChabochePlasticHardening
    back_stress = 'X3'
    C = 10000
    g = 100
    A = 1e-8
    a = 1.2
  []
  [X4rate]
    type = ChabochePlasticHardening
    back_stress = 'X4'
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
    variable = 'strain'
  []
  [Eerate]
    type = SR2LinearCombination
    from = 'strain_rate plastic_strain_rate'
    to = 'elastic_strain_rate'
    weights = '1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
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
  [integrate_X3]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X3'
  []
  [integrate_X4]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X4'
  []
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'stress'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'isoharden kinharden mandel_stress overstress vonmises yield_surface normality
              flow_rate eprate Eprate X1rate X2rate X3rate X4rate Erate Eerate elasticity
              integrate_stress integrate_ep integrate_X1 integrate_X2 integrate_X3 integrate_X4'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'stress equivalent_plastic_strain X1 X2 X3 X4'
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
    unknowns_SR2 = 'stress X1 X2 X3 X4'
    unknowns_Scalar = 'equivalent_plastic_strain'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
[]
