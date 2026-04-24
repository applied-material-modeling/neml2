## Applying KKT conditions with the help of Fisher-Burmeister complementary condition

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'E'
    force_SR2_values = 'strains'
    predictor = LINEAR_EXTRAPOLATION
    save_as = 'fb_pff_result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/pff_result.pt'
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
    values = 0.0152 # Gc/l/Co with Gc = 95 N/m, l = 3.125 mm, Co = 2
  []
[]

[Models]
  # strain energy density: g * psie0
  [degrade]
    type = PowerDegradationFunction
    phase = 'd'
    degradation = 'g'
    power = 'p'
  []
  [sed0]
    type = LinearIsotropicStrainEnergyDensity
    strain = 'E'
    active_strain_energy_density = 'psie_active'
    inactive_strain_energy_density = 'psie_inactive'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    coefficients = '25.84e3 0.18'
    # decomposition = 'NONE'
    decomposition = 'VOLDEV'
  []
  [sed1]
    type = ScalarMultiplication
    from = 'g psie_active'
    to = 'psie_degraded'
  []
  [sed]
    type = ScalarLinearCombination
    from = 'psie_degraded psie_inactive'
    to = 'psie'
    weights = '1 1'
  []
  # crack geometric function: alpha
  [cracked]
    type = CrackGeometricFunctionAT2
    phase = 'd'
    crack = 'alpha'
  []
  # total energy
  [sum]
    type = ScalarLinearCombination
    from = 'alpha psie'
    to = 'psi'
    weights = 'GcbylbyCo 1'
  []
  [energy] # this guy maps from (strain, d) -> energy
    type = ComposedModel
    models = 'degrade sed0 sed1 sed cracked sum'
  []
  # phase rate, follows from variation of total energy w.r.t. phase field
  [dpsidd]
    type = Normality
    model = 'energy'
    function = 'psi'
    from = 'd'
    to = 'dpsi_dd'
  []
  # obtain d_rate
  [drate]
    type = ScalarVariableRate
    variable = 'd'
  []
  # define functional
  [functional]
    type = ScalarLinearCombination
    from = 'dpsi_dd d_rate'
    to = 'F'
    weights = '1 1'
  []
  # Fisher Burmeister Complementary Condition
  [Fish_Burm]
    type = FischerBurmeister
    a = 'F'
    b = 'd_rate'
    complementarity = 'd_residual'
    a_inequality = 'LE'
  []
  # system of equations
  [eq]
    type = ComposedModel
    models = 'Fish_Burm functional drate dpsidd'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'eq'
    unknowns = 'd'
  []
[]

[Solvers]
  [newton]
    type = Newton
    rel_tol = 1e-08
    abs_tol = 1e-10
    max_its = 50
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  # solve for d
  [solve_d]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
  # after the solve take derivative of the total energy w.r.t. strain to get stress
  [stress]
    type = Normality
    model = 'energy'
    function = 'psi'
    from = 'E'
    to = 'S'
  []
  [model]
    type = ComposedModel
    models = 'solve_d stress'
    additional_outputs = 'd'
  []
[]
