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
  # end_time = LogspaceScalar(3, 3, nbatch) -> shape (nbatch,), constant 1e3
  [end_time]
    type = Python
    expr = '''
      Scalar(torch.logspace(3.0, 3.0, nbatch, dtype=torch.float64))
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
  # start_temperature = LinspaceScalar(300, 300, nbatch) -> constant 300, (nbatch,)
  [start_temperature]
    type = Python
    expr = '''
      Scalar(torch.linspace(300.0, 300.0, nbatch, dtype=torch.float64))
    '''
  []
  # end_temperature = LinspaceScalar(1800, 1800, nbatch) -> constant 1800, (nbatch,)
  [end_temperature]
    type = Python
    expr = '''
      Scalar(torch.linspace(1800.0, 1800.0, nbatch, dtype=torch.float64))
    '''
  []
  # temperatures = LinspaceScalar(start_temperature, end_temperature, 100) -> (100, nbatch)
  [temperatures]
    type = Python
    expr = '''
      Scalar(
          start_temperature.data.unsqueeze(0)
          + (end_temperature.data - start_temperature.data).unsqueeze(0)
          * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1)
      )
    '''
  []
  # max_strain = FillSR2(exx=0, eyy=0, ezz=0) broadcast to (nbatch, 6) -- all zeros
  [max_strain]
    type = Python
    expr = '''
      SR2.fill(0.0, 0.0, 0.0).dynamic_batch.expand(nbatch)
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
  # f0 = Scalar(0.36)
  [f0]
    type = Python
    expr = '''
      Scalar(torch.tensor([0.36], dtype=torch.float64))
    '''
  []
  # gamma = surface_tension scalar. The v2 benchmark made this per-batch
  # (`LinspaceScalar(0, 150, nbatch)`) to sweep surface tension across batch
  # elements; that pattern bakes a rank-1 parameter and conflicts with the
  # dynamic-batch artifact (see tests/aoti/static_batch_baked/ for the
  # `dynamic_batch=false` counterpart that exercises it). Held scalar here
  # so the benchmark stays cheap and batch-generalizable.
  [gamma]
    type = Python
    expr = '''
      Scalar(torch.tensor(75.0, dtype=torch.float64))
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
    force_Scalar_names = 'temperature'
    force_Scalar_values = 'temperatures'
    ic_Scalar_names = 'void_fraction'
    ic_Scalar_values = 'f0'
  []
[]

[Models]
  [isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 5
    saturation_rate = 1.2
  []
  [sintering_stress]
    type = OlevskySinteringStress
    surface_tension = 'gamma'
    particle_radius = 3e-4
  []
  [eigenstrain]
    type = ThermalEigenstrain
    reference_temperature = 300
    CTE = 1e-6
  []
  [elastic_strain]
    type = SR2LinearCombination
    from = 'strain plastic_strain eigenstrain'
    to = 'elastic_strain'
    weights = '1 -1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '3e4 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
  []
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'stress'
  []
  [j2]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'mandel_stress'
    invariant = 'flow_invariant'
  []
  [sh]
    type = SR2Invariant
    invariant_type = 'I1'
    tensor = 'mandel_stress'
    invariant = 'hydrostatic_stress'
  []
  [sp]
    type = ScalarLinearCombination
    from = 'hydrostatic_stress sintering_stress'
    to = 'poro_invariant'
    weights = '1 -1'
  []
  [q1]
    type = ArrheniusParameter
    temperature = 'temperature'
    reference_value = 8000
    activation_energy = 5e4
    ideal_gas_constant = 8.314
  []
  [yield_surface]
    type = GTNYieldFunction
    yield_stress = 60.0
    q1 = 'q1'
    q2 = 0.01
    q3 = 1.57
    isotropic_hardening = 'isotropic_hardening'
  []
  [flow]
    type = ComposedModel
    models = 'j2 sh sp yield_surface'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 500
    exponent = 2
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'yield_function'
    from = 'mandel_stress isotropic_hardening'
    to = 'flow_direction isotropic_hardening_direction'
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [voidrate]
    type = GursonCavitation
  []
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'plastic_strain'
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
  []
  [integrate_void]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'void_fraction'
  []
  [surface]
    type = ComposedModel
    models = 'isoharden sintering_stress elastic_strain elasticity
              mandel_stress flow flow_rate
              normality
              Eprate eprate voidrate
              integrate_Ep integrate_ep integrate_void'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'surface'
    unknowns = 'plastic_strain equivalent_plastic_strain void_fraction'
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
    unknowns_SR2 = 'plastic_strain'
    unknowns_Scalar = 'equivalent_plastic_strain void_fraction'
  []
  [return_map]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model]
    type = ComposedModel
    models = 'eigenstrain return_map elastic_strain elasticity'
    additional_outputs = 'plastic_strain equivalent_plastic_strain void_fraction'
  []
[]
