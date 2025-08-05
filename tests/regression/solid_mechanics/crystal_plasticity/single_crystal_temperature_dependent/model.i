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
  [dxx]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.1
  []
  [dyy]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.05
  []
  [dzz]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.05
  []
  [deformation_rate_single]
    type = FillSR2
    values = 'dxx dyy dzz'
  []
  [deformation_rate]
    type = LinspaceSR2
    start = deformation_rate_single
    end = deformation_rate_single
    nstep = 100
  []

  [w1]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.1
  []
  [w2]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.05
  []
  [w3]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.05
  []
  [vorticity_single]
    type = FillWR2
    values = 'w1 w2 w3'
  []
  [vorticity]
    type = LinspaceWR2
    start = vorticity_single
    end = vorticity_single
    nstep = 100
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

  [T_control]
    type = Scalar
    values = '0 100'
    batch_shape = '(2)'
  []
  [E_values]
    type = Scalar
    values = '1e5 1e6'
    batch_shape = '(2)'
  []
  [nu_values]
    type = Scalar
    values = '0.25 0.3'
    batch_shape = '(2)'
  []
  [n_values]
    type = Scalar
    values = '8.0 10.0'
    batch_shape = '(2)'
  []


  [T_start]
    type = LinspaceScalar
    start = 1.0
    end = 2.0
    nstep = 20
  []
  [T_end]
    type = LinspaceScalar
    start = 98.0
    end = 99.0
    nstep = 20
  []
  [temperature]
    type = LinspaceScalar
    start = T_start
    end = T_end
    nstep = 100
  []
[]

[Drivers]
  [driver]
    type = LDISolidMechanicsDriver
    model = 'model_with_stress'
    prescribed_time = 'times'
    prescribed_deformation_rate = 'deformation_rate'
    prescribed_vorticity = 'vorticity'
    prescribed_temperature = 'temperature'
    ic_Rot_names = 'state/orientation'
    ic_Rot_values = 'initial_orientation'
    predictor = 'PREVIOUS_STATE'
    cp_warmup = true
    cp_warmup_elastic_scale = 0.1
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
    type = NewtonWithLineSearch
    max_linesearch_iterations = 5
  []
[]

[Data]
  [crystal_geometry]
    type = CubicCrystal
    lattice_parameter = "a"
    slip_directions = "sdirs"
    slip_planes = "splanes"
  []
[]

[Models]
  [euler_rodrigues]
    type = RotationMatrix
    from = 'state/orientation'
    to = 'state/orientation_matrix'
  []
  [E]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = 'T_control'
    ordinate = 'E_values'
  []
  [nu]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = 'T_control'
    ordinate = 'nu_values'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = 'E nu'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = "state/elastic_strain"
    stress = "state/internal/cauchy_stress"
  []
  [resolved_shear]
    type = ResolvedShear
  []
  [elastic_stretch]
    type = ElasticStrainRate
  []
  [plastic_spin]
    type = PlasticVorticity
  []
  [plastic_deformation_rate]
    type = PlasticDeformationRate
  []
  [orientation_rate]
    type = OrientationRate
  []
  [sum_slip_rates]
    type = SumSlipRates
  []
  [n]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = 'T_control'
    ordinate = 'n_values'
  []
  [slip_rule]
    type = PowerLawSlipRule
    n = 'n'
    gamma0 = 1.0e-1
  []
  [slip_strength]
    type = SingleSlipStrengthMap
    constant_strength = 100.0
  []
  [voce_hardening]
    type = VoceSingleSlipHardeningRule
    initial_slope = 1000.0
    saturated_hardening = 100.0
  []
  [integrate_slip_hardening]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/internal/slip_hardening'
  []
  [integrate_elastic_strain]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/elastic_strain'
  []
  [integrate_orientation]
    type = WR2ImplicitExponentialTimeIntegration
    variable = 'state/orientation'
  []

  [implicit_rate]
    type = ComposedModel
    models = "euler_rodrigues elasticity orientation_rate resolved_shear
              elastic_stretch plastic_deformation_rate plastic_spin
              sum_slip_rates slip_rule slip_strength voce_hardening
              integrate_slip_hardening integrate_elastic_strain integrate_orientation"
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
  [model_with_stress]
    type = ComposedModel
    models = 'model elasticity'
    additional_outputs = 'state/elastic_strain'
  []
[]
