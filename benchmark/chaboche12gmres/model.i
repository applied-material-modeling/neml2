# neml2
# chaboche12gmres: 12-backstress Chaboche viscoplasticity (U=79, a large DENSE
# single-group implicit system) -- the iterative-solver scaling benchmark.
# Load history is the first 10 steps of the standard chaboche sweep (same
# per-step increment, 1/10 the wall-clock) so direct vs iterative timing is
# cheap to measure. This dir = the matrix-free GMRES case; chaboche12 (DenseLU)
# is the DIRECT baseline (identical model, [Solvers] swapped to DenseLU).
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
  # times = LinspaceScalar(0, end_time*9/99, 10)  # first 10 of the 100-step sweep -> shape (10, nbatch)
  [times]
    type = Python
    expr = '''
      result = Scalar(
          end_time.data.unsqueeze(0)
          * torch.linspace(0.0, 9.0 / 99.0, 10, dtype=torch.float64).unsqueeze(-1)
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
  # strains = LinspaceSR2(0, max_strain*9/99, 10)  # first 10 of the 100-step sweep -> shape (10, nbatch, 6)
  [strains]
    type = Python
    expr = '''
      SR2(
          max_strain.data.unsqueeze(0)
          * torch.linspace(0.0, 9.0 / 99.0, 10, dtype=torch.float64).reshape(10, 1, 1)
      )
    '''
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_SR2_names = 'strain'
    prescribed_SR2_values = 'strains'
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
    from = 'X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12'
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
  [X5rate]
    type = ChabochePlasticHardening
    back_stress = 'X5'
    C = 10000
    g = 100
    A = 1e-8
    a = 1.2
  []
  [X6rate]
    type = ChabochePlasticHardening
    back_stress = 'X6'
    C = 1000
    g = 9
    A = 1e-10
    a = 3.2
  []
  [X7rate]
    type = ChabochePlasticHardening
    back_stress = 'X7'
    C = 10000
    g = 100
    A = 1e-8
    a = 1.2
  []
  [X8rate]
    type = ChabochePlasticHardening
    back_stress = 'X8'
    C = 1000
    g = 9
    A = 1e-10
    a = 3.2
  []
  [X9rate]
    type = ChabochePlasticHardening
    back_stress = 'X9'
    C = 10000
    g = 100
    A = 1e-8
    a = 1.2
  []
  [X10rate]
    type = ChabochePlasticHardening
    back_stress = 'X10'
    C = 1000
    g = 9
    A = 1e-10
    a = 3.2
  []
  [X11rate]
    type = ChabochePlasticHardening
    back_stress = 'X11'
    C = 10000
    g = 100
    A = 1e-8
    a = 1.2
  []
  [X12rate]
    type = ChabochePlasticHardening
    back_stress = 'X12'
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
  [integrate_X5]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X5'
  []
  [integrate_X6]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X6'
  []
  [integrate_X7]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X7'
  []
  [integrate_X8]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X8'
  []
  [integrate_X9]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X9'
  []
  [integrate_X10]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X10'
  []
  [integrate_X11]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X11'
  []
  [integrate_X12]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X12'
  []
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'stress'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'isoharden kinharden mandel_stress overstress vonmises yield_surface normality
              flow_rate eprate Eprate X1rate X2rate X3rate X4rate X5rate X6rate X7rate X8rate X9rate X10rate X11rate X12rate Erate Eerate elasticity
              integrate_stress integrate_ep integrate_X1 integrate_X2 integrate_X3 integrate_X4 integrate_X5 integrate_X6 integrate_X7 integrate_X8 integrate_X9 integrate_X10 integrate_X11 integrate_X12'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'stress equivalent_plastic_strain X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12'
  []
[]

[Solvers]
  [newton]
    type = Newton
    linear_solver = 'gmres'
  []
  [gmres]
    type = GMRES
  []
[]

[Models]
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_SR2 = 'stress X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12'
    unknowns_Scalar = 'equivalent_plastic_strain'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
[]
