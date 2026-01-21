[Drivers]
  [driver]
    type = SDTSolidMechanicsDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_strain = 'strains'
    save_as = 'result.pt'
  []
[]

[Tensors]
  [times]
    type = LinspaceScalar
    start = 0
    end = 1
    nstep = 20
  []
  [exx]
    type = FullScalar
    value = 0.01
  []
  [eyy]
    type = FullScalar
    value = -0.005
  []
  [ezz]
    type = FullScalar
    value = -0.005
  []
  [max_strain]
    type = FillSR2
    values = 'exx eyy ezz'
  []
  [strains]
    type = LinspaceSR2
    start = 0
    end = max_strain
    nstep = 20
  []
[]

[Models]
  [eq1]
    type = SR2LinearCombination
    from_var = 'forces/E state/Ep'
    to_var = 'state/Ee'
    coefficients = '1 -1'
  []
  [eq2]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'state/Ee'
    stress = 'state/S'
  []
  [eq3]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/S'
    invariant = 'state/s'
  []
  [eq4]
    type = YieldFunction
    yield_stress = 5
    yield_function = 'state/fp'
    effective_stress = 'state/s'
  []
  [surface]
    type = ComposedModel
    models = 'eq3 eq4'
  []
  [eq5]
    type = Normality
    model = 'surface'
    function = 'state/fp'
    from = 'state/S'
    to = 'state/N'
  []
  [eq6]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 2
    yield_function = 'state/fp'
    flow_rate = 'state/gamma_rate'
  []
  [eq7]
    type = AssociativePlasticFlow
    flow_rate = 'state/gamma_rate'
    flow_direction = 'state/N'
    plastic_strain_rate = 'state/Ep_rate'
  []
  [eq8]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/Ep'
  []
  [system]
    type = ComposedModel
    models = 'eq1 eq2 surface eq5 eq6 eq7 eq8'
  []
[]

[EquationSystems]
  [eq_sys]
    type = ModelEquationSystem
    model = 'system'
  []
[]

[Solvers]
  [newton]
    type = Newton
    rel_tol = 1e-08
    abs_tol = 1e-10
    max_its = 50
    verbose = true
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [model0]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
  [model]
    type = ComposedModel
    models = 'model0 eq1 eq2'
    additional_outputs = 'state/Ep'
  []
[]
