
## Applying KKT conditions with the help of Fisher-Burmeister complementary condition

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'forces/E'
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

[Solvers]
  [newton]
    type = Newton
    rel_tol = 1e-08
    abs_tol = 1e-10
    max_its = 50
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
    phase = 'state/d'
    degradation = 'state/g'
    power = 'p'
  []
  [sed0]
    type = LinearIsotropicStrainEnergyDensity
    strain = 'forces/E'
    strain_energy_density_active = 'state/psie_active'
    strain_energy_density_inactive = 'state/psie_inactive'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    coefficients = '25.84e3 0.18'
    # decomposition = 'NONE'
    decomposition = 'VOLDEV'
  []
  [sed1]
    type = ScalarMultiplication
    from_var = 'state/g state/psie_active'
    to_var = 'state/psie_degraded'
  []
  [sed]
    type = ScalarLinearCombination
    from_var = 'state/psie_degraded state/psie_inactive'
    to_var = 'state/psie'
    coefficients = '1 1'
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
    coefficients = 'GcbylbyCo 1'
  []
  [energy] # this guy maps from (strain, d) -> energy
    type = ComposedModel
    models = 'degrade sed0 sed1 sed cracked sum'
  []
  # phase rate, follows from variation of total energy w.r.t. phase field
  [dpsidd]
    type = Normality
    model = 'energy'
    function = 'state/psi'
    from = 'state/d'
    to = 'state/dpsi_dd'
  []
  # obtain d_rate
  [drate]
    type = ScalarVariableRate
    variable = 'state/d'
    rate = 'state/d_rate'
  []
  # define functional
  [functional]
    type = ScalarLinearCombination
    from_var = 'state/dpsi_dd state/d_rate'
    to_var = 'state/F'
    coefficients = '1 1'
  []
  # Fisher Burmeister Complementary Condition
  [Fish_Burm]
    type = FischerBurmeister
    first_var = 'state/F'
    second_var = 'state/d_rate'
    fischer_burmeister = 'residual/d'

  []
  # system of equations
  [eq]
    type = ComposedModel
    models = 'Fish_Burm functional drate dpsidd'
  []
  # solve for d
  [solve_d]
    type = ImplicitUpdate
    implicit_model = 'eq'
    solver = 'newton'
  []
  # after the solve take derivative of the total energy w.r.t. strain to get stress
  [stress]
    type = Normality
    model = 'energy'
    function = 'state/psi'
    from = 'forces/E'
    to = 'state/S'
  []
  [model]
    type = ComposedModel
    models = 'solve_d stress'
    additional_outputs = 'state/d'
  []
[]
