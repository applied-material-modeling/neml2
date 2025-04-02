[Tensors]
  [end_time]
    type = LinspaceScalar
    start = 1
    end = 10
    nstep = 20
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
    nstep = 20
  []
  [F]
    type = LinspaceR2
    start = F_start
    end = F_end
    nstep = 100
  []
  [Fp0]
    type = FillR2
    values = '1 1 1'
  []

  [a]
    type = Scalar
    values = '1.0'
  []
  [sdirs]
    type = FillMillerIndex
    values = '1 1 0'
  []
  [splanes]
    type = FillMillerIndex
    values = '1 1 1'
  []

  [R1]
    type = LinspaceScalar
    start = 0
    end = 0.75
    nstep = 20
  []
  [R2]
    type = LinspaceScalar
    start = 0
    end = -0.25
    nstep = 20
  []
  [R3]
    type = LinspaceScalar
    start = -0.1
    end = 0.1
    nstep = 20
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
    force_R2_names = 'forces/F'
    force_R2_values = 'F'
    force_Rot_names = 'forces/r'
    force_Rot_values = 'r'
    ic_R2_names = 'state/Fp'
    ic_R2_values = 'Fp0'
    predictor = 'LINEAR_EXTRAPOLATION'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Solvers]
  [newton]
    type = Newton
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
    from = 'forces/r'
    to = 'forces/R'
  []
  # Hardening (this is just a very simple hardening model)
  [slip_strength]
    type = SingleSlipStrengthMap
    constant_strength = 50.0
    slip_hardening = 'state/tauc'
    slip_strengths = 'state/tauc_i'
  []
  [voce_hardening]
    type = VoceSingleSlipHardeningRule
    initial_slope = 500.0
    saturated_hardening = 50.0
    slip_hardening_rate = 'state/tauc_rate'
    slip_hardening = 'state/tauc'
    sum_slip_rates = 'state/gamma_rate'
  []
  # Elasticity: St. Venant-Kirchhoff with Green-Lagrange strain
  [mult_decomp]
    type = R2Multiplication
    A = 'forces/F'
    B = 'state/Fp'
    to = 'state/Fe'
    invert_B = true
  []
  [gl_strain]
    type = GreenLagrangeStrain
    deformation_gradient = 'state/Fe'
    strain = 'state/E'
  []
  [svk]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.25'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'state/E'
    stress = 'state/S'
  []
  [elasticity]
    type = ComposedModel
    models = 'mult_decomp gl_strain svk'
  []
  # CP flow rule
  [resolved_shear]
    type = ResolvedShear
    resolved_shears = 'state/tau_i'
    stress = 'state/S'
    orientation = 'forces/R'
  []
  [slip_rule]
    type = PowerLawSlipRule
    n = 8.0
    gamma0 = 2.0e-1
    slip_rates = 'state/gamma_rate_i'
    resolved_shears = 'state/tau_i'
    slip_strengths = 'state/tauc_i'
  []
  [sum_slip_rates]
    type = SumSlipRates
    slip_rates = 'state/gamma_rate_i'
    sum_slip_rates = 'state/gamma_rate'
  []
  [plastic_velgrad_sym]
    type = PlasticDeformationRate
    plastic_deformation_rate = 'state/Lp_sym'
    slip_rates = 'state/gamma_rate_i'
    orientation = 'forces/R'
  []
  [plastic_velgrad]
    type = SR2toR2
    input = 'state/Lp_sym'
    output = 'state/Lp'
  []
  [plastic_defgrad_rate]
    type = R2Multiplication
    A = 'state/Lp'
    B = 'state/Fp'
    to = 'state/Fp_rate'
  []
  # Definition of residuals
  [integrate_slip_hardening]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/tauc'
  []
  [integrate_plastic_defgrad]
    type = R2BackwardEulerTimeIntegration
    variable = 'state/Fp'
  []
  # The implicit model that we solve for
  [implicit_rate]
    type = ComposedModel
    models = "euler_rodrigues slip_strength voce_hardening
              mult_decomp gl_strain svk
              resolved_shear slip_rule sum_slip_rates
              plastic_velgrad_sym plastic_velgrad plastic_defgrad_rate
              integrate_slip_hardening integrate_plastic_defgrad"
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
  [model_with_pk2_stress]
    type = ComposedModel
    models = 'model elasticity'
    additional_outputs = 'state/Fp'
  []
[]
