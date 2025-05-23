## Applying the phase field driving force in the form of Allen-Cahn equation

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'forces/E'
    force_SR2_values = 'strains'
    predictor = LINEAR_EXTRAPOLATION
    save_as = 'pff_result.pt'
    show_input_axis = true
    show_output_axis = true
    verbose = true
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/pff_result.pt'
  []
[]

[Solvers]
  [newton]
    type = Newton
    rel_tol = 1e-08
    abs_tol = 1e-10
    max_its = 50
    verbose = true
  []
[]

[Tensors]
  [times]
    type = LinspaceScalar
    start = 0
    end = 3
    nstep = 1000
  []
  [exx]
    type = FullScalar
    value = 0.016
  []
  [eyy]
    type = FullScalar
    value = -0.008
  []
  [ezz]
    type = FullScalar
    value = -0.008
  []
  [max_strain]
    type = FillSR2
    values = 'exx eyy ezz'
  []
  [strains]
    type = LinspaceSR2
    start = 0
    end = max_strain
    nstep = 1000
  []
  [p]
    type = Scalar
    values = 2
  []
  [GcbylbyCo]
    type = Scalar
    values = -0.0152 # Gc/l/Co with Gc = 95 N/m, l = 3.125 mm, Co = 2, -ve sign to match the AC eqn
  []
[]

[Models]
  # strain energy density: g * psie0
  [degrade]
    type = PowerDegradationFunction
    phase = 'state/d'
    degradation = 'state/g'
    power = 'p'
  []
  [sed0]
    type = LinearIsotropicStrainEnergyDensity
    strain = 'forces/E'
    strain_energy_density = 'state/psie0'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    coefficients = '25.84e3 0.18'
  []
  [sed]
    type = ScalarMultiplication
    from_var = 'state/g state/psie0'
    to_var = 'state/psie'
  []
  # crack geometric function: alpha
  [cracked]
    type = CrackGeometricFunctionAT2
    phase = 'state/d'
    crack = 'state/alpha'
  []
  # total energy
  [sum]
    type = ScalarLinearCombination
    from_var = 'state/alpha state/psie'
    to_var = 'state/psi'
    coefficients = 'GcbylbyCo -1'  # -ve sign to match the AC eqn
  []
  [energy] # this guy maps from (strain, d) -> energy
    type = ComposedModel
    models = 'degrade sed0 sed cracked sum'
  []
  # phase rate, follows from variation of total energy w.r.t. phase field
  [phase_rate]
    type = Normality
    model = 'energy'
    function = 'state/psi'
    from = 'state/d'
    to = 'state/d_rate'
  []
  # integrate d
  [integrate_d]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/d'
  []
  # solve the equation
  [eq]
    type = ComposedModel
    models = 'phase_rate integrate_d'
  []
  [solve_d]
    type = ImplicitUpdate
    implicit_model = 'eq'
    solver = 'newton'
  []
  # after the solve, re-evaluate the elastic strain energy, then take derivative w.r.t. strain to get stress
  [evaluate_sed]
    type = ComposedModel
    models = 'degrade sed0 sed'
  []
  [stress]
    type = Normality
    model = 'evaluate_sed'
    function = 'state/psie'
    from = 'forces/E'
    to = 'state/S'
  []
  [model]
    type = ComposedModel
    models = 'solve_d stress'
    additional_outputs = 'state/d'
  []
[]