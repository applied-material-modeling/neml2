# krylov_gmres_precond: same DENSE implicit return as krylov_gmres, but the GMRES
# solver uses a Block-Jacobi preconditioner with the chord cache strategy. This
# exercises the compiled PRECONDITIONED Krylov path -- the preconditioner is an
# authored [Solvers] object (BlockJacobiPreconditioner) referenced by GMRES, so
# the artifact carries the precond_setup / precond_apply graphs (the C++ holds the
# returned state and caches it across Newton iterations per the chord strategy).
# Auto-discovered by test_aoti.py for py-eager == py-aoti/cpp-aoti parity.

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
  [yield_surface]
    type = YieldFunction
    yield_stress = 5
  []
  [flow]
    type = ComposedModel
    models = 'vonmises yield_surface'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'yield_function'
    from = 'mandel_stress'
    to = 'flow_direction'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 2
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
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'stress'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'mandel_stress vonmises yield_surface normality flow_rate Eprate Erate Eerate elasticity integrate_stress'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'stress'
    residuals = 'stress_residual'
  []
[]

[Solvers]
  [newton]
    type = Newton
    linear_solver = 'gmres'
  []
  [gmres]
    type = GMRES
    preconditioner = 'bjac'
    cache_strategy = 'chord'
  []
  [bjac]
    type = BlockJacobiPreconditioner
  []
[]

[Models]
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_SR2 = 'stress'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
[]
