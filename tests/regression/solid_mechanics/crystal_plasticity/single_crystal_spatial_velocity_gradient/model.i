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
  [lxx]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.1
  []
  [lyy]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.05
  []
  [lzz]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.05
  []

  [lxy]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.01
  []
  [lxz]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.02
  []
  [lyx]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.01
  []
  [lyz]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.025
  []
  [lzx]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.03
  []
  [lzy]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.01
  []

  [l_single]
    type = FillR2
    values = 'lxx lxy lxz lyx lyy lyz lzx lzy lzz'
  []
  [L]
    type = LinspaceR2
    start = l_single
    end = l_single
    nstep = 100
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
[]

[Drivers]
  [driver]
    type = SpatialVelocityDriver
    model = 'model_with_stress'
    prescribed_time = 'times'
    prescribed_spatial_velocity_gradient = 'L'
    ic_Rot_names = 'state/orientation'
    ic_Rot_values = 'initial_orientation'
    predictor = 'PREVIOUS_STATE'
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
    lattice_parameter = "a"
    slip_directions = "sdirs"
    slip_planes = "splanes"
  []
[]

[Models]
  [split_to_deformation_rate]
    type = R2toSR2
    input = 'forces/spatial_velocity_gradient'
    output = 'forces/deformation_rate'
  []
  [split_to_vorticity]
    type = R2toWR2
    input = 'forces/spatial_velocity_gradient'
    output = 'forces/vorticity'
  []
  [euler_rodrigues]
    type = RotationMatrix
    from = 'state/orientation'
    to = 'state/orientation_matrix'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    coefficients = '1e5 0.25'
    strain = 'state/elastic_strain'
    stress = 'state/internal/cauchy_stress'
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
  [slip_rule]
    type = PowerLawSlipRule
    n = 8.0
    gamma0 = 2.0e-1
  []
  [slip_strength]
    type = SingleSlipStrengthMap
    constant_strength = 50.0
  []
  [voce_hardening]
    type = VoceSingleSlipHardeningRule
    initial_slope = 500.0
    saturated_hardening = 50.0
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
    models = "split_to_deformation_rate split_to_vorticity
              euler_rodrigues elasticity orientation_rate resolved_shear
              elastic_stretch plastic_deformation_rate plastic_spin
              sum_slip_rates slip_rule slip_strength voce_hardening
              integrate_slip_hardening integrate_elastic_strain integrate_orientation"
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
  []
[]

[Solvers]
  [newton]
    type = NewtonWithLineSearch
    max_linesearch_iterations = 5
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
  [full_stress]
    type = SR2toR2
    input = 'state/internal/cauchy_stress'
    output = 'state/internal/full_cauchy_stress'
  []
  [model_with_stress]
    type = ComposedModel
    models = 'model elasticity full_stress'
    additional_outputs = 'state/elastic_strain'
  []
[]
