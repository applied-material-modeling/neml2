nbatch = 20

[Tensors]
  [end_time]
    type = LinspaceScalar
    start = 1
    end = 10
    nstep = ${nbatch}
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = end_time
    nstep = 100
  []
  [F_start]
    type = FillR2
    values = '1 0 0 0 1 0 0 0 1'
  []
  [F_end_min]
    type = FillR2
    values = '1.005 0.001 0.005 0.001 0.991 -0.03 -0.005 0.002 1.008'
  []
  [F_end_max]
    type = FillR2
    values = '1.05 0.01 0.05 0.01 0.91 -0.3 -0.05 0.02 1.08'
  []
  [F_end]
    type = LinspaceR2
    start = F_end_min
    end = F_end_max
    nstep = ${nbatch}
  []
  [F]
    type = LinspaceR2
    start = F_start
    end = F_end
    nstep = 100
  []
  [Fp0]
    type = R2
    values = '1 0 0 0 1 0 0 0 1'
  []

  [a]
    type = Scalar
    values = '1.0'
  []
  [sdirs]
    type = MillerIndex
    values = '1 1 0'
  []
  [splanes]
    type = MillerIndex
    values = '1 1 1'
  []

  [R1]
    type = LinspaceScalar
    start = 0
    end = 0.75
    nstep = ${nbatch}
  []
  [R2]
    type = LinspaceScalar
    start = 0
    end = -0.25
    nstep = ${nbatch}
  []
  [R3]
    type = LinspaceScalar
    start = -0.1
    end = 0.1
    nstep = ${nbatch}
  []

  [initial_orientation]
    type = FillRot
    values = 'R1 R2 R3'
    method = 'standard'
  []
  [r]
    type = LinspaceRot
    start = initial_orientation
    end = initial_orientation
    nstep = 100
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model_with_pk2_stress'
    prescribed_time = 'times'
    force_R2_names = 'F'
    force_R2_values = 'F'
    force_Rot_names = 'r'
    force_Rot_values = 'r'
    ic_R2_names = 'Fp'
    ic_R2_values = 'Fp0'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Data]
  [crystal_geometry]
    type = CubicCrystal
    lattice_parameter = 'a'
    slip_directions = 'sdirs'
    slip_planes = 'splanes'
  []
[]

[Models]
  # Orientation remains constant as we work with the reference configuration
  [euler_rodrigues]
    type = RotationMatrix
    from = 'r'
    to = 'R'
  []
  # Hardening (this is just a very simple hardening model)
  [slip_strength]
    type = SingleSlipStrengthMap
    constant_strength = 50.0
    slip_hardening = 'tauc'
    slip_strengths = 'tauc_i'
  []
  [voce_hardening]
    type = VoceSingleSlipHardeningRule
    initial_slope = 500.0
    saturated_hardening = 50.0
    slip_hardening = 'tauc'
    sum_slip_rates = 'gamma_rate'
  []
  # Elasticity: St. Venant-Kirchhoff with Green-Lagrange strain
  [mult_decomp]
    type = R2Multiplication
    A = 'F'
    B = 'Fp'
    to = 'Fe'
    invert_B = true
  []
  [gl_strain]
    type = GreenLagrangeStrain
    deformation_gradient = 'Fe'
    strain = 'E'
  []
  [svk]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.25'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'E'
    stress = 'S'
  []
  [elasticity]
    type = ComposedModel
    models = 'mult_decomp gl_strain svk'
  []
  # CP flow rule
  [resolved_shear]
    type = ResolvedShear
    resolved_shears = 'tau_i'
    stress = 'S'
    orientation_matrix = 'R'
  []
  [slip_rule]
    type = PowerLawSlipRule
    n = 8.0
    gamma0 = 2.0e-1
    slip_rates = 'gamma_rate_i'
    resolved_shears = 'tau_i'
    slip_strengths = 'tauc_i'
  []
  [sum_slip_rates]
    type = SumSlipRates
    slip_rates = 'gamma_rate_i'
    sum_slip_rates = 'gamma_rate'
  []
  [plastic_velgrad]
    type = PlasticSpatialVelocityGradient
    plastic_spatial_velocity_gradient = 'Lp'
    slip_rates = 'gamma_rate_i'
    orientation_matrix = 'R'
  []
  [plastic_defgrad_rate]
    type = R2Multiplication
    A = 'Lp'
    B = 'Fp'
    to = 'Fp_rate'
  []
  # Definition of residuals
  [integrate_slip_hardening]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'tauc'
  []
  [integrate_plastic_defgrad]
    type = R2BackwardEulerTimeIntegration
    variable = 'Fp'
  []
  [implicit_rate]
    type = ComposedModel
    models = "euler_rodrigues slip_strength voce_hardening
              elasticity resolved_shear slip_rule sum_slip_rates
              plastic_velgrad plastic_defgrad_rate
              integrate_slip_hardening integrate_plastic_defgrad"
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'tauc Fp'
    residuals = 'tauc_residual Fp_residual'
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
    unknowns_Scalar = 'tauc'
    unknowns_R2 = 'Fp'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model_with_pk2_stress]
    type = ComposedModel
    models = 'model elasticity'
    additional_outputs = 'Fp'
  []
[]
