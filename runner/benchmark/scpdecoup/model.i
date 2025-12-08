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
  [dxx]
    type = FullScalar
    batch_shape = '(${nbatch})'
    value = 0.1
  []
  [dyy]
    type = FullScalar
    batch_shape = '(${nbatch})'
    value = -0.05
  []
  [dzz]
    type = FullScalar
    batch_shape = '(${nbatch})'
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
    batch_shape = '(${nbatch})'
    value = 0.1
  []
  [w2]
    type = FullScalar
    batch_shape = '(${nbatch})'
    value = -0.05
  []
  [w3]
    type = FullScalar
    batch_shape = '(${nbatch})'
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
[]

[Drivers]
  [driver]
    type = LDISolidMechanicsDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_deformation_rate = 'deformation_rate'
    prescribed_vorticity = 'vorticity'
    ic_Rot_names = 'state/orientation'
    ic_Rot_values = 'initial_orientation'
    predictor = 'PREVIOUS_STATE'
    cp_warmup = true
    cp_warmup_elastic_scale = 0.1
    device = ${device}
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

[Data]
  [crystal_geometry]
    type = CubicCrystal
    lattice_parameter = 'a'
    slip_directions = 'sdirs'
    slip_planes = 'splanes'
  []
[]

[Models]
  ############################################################################
  # Sub-system #1 for updating elastic strain and internal variables
  ############################################################################
  [euler_rodrigues_1]
    type = RotationMatrix
    from = 'forces/tmp/orientation'
    to = 'state/orientation_matrix'
  []
  [elasticity_1]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.25'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'state/elastic_strain'
    stress = 'state/internal/cauchy_stress'
  []
  [resolved_shear]
    type = ResolvedShear
  []
  [elastic_stretch]
    type = ElasticStrainRate
  []
  [plastic_deformation_rate]
    type = PlasticDeformationRate
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
  [implicit_rate_1]
    type = ComposedModel
    models = "euler_rodrigues_1 elasticity_1 resolved_shear
              elastic_stretch plastic_deformation_rate
              sum_slip_rates slip_rule slip_strength voce_hardening
              integrate_slip_hardening integrate_elastic_strain"
  []
  [subsystem1]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate_1'
    solver = 'newton'
  []

  ############################################################################
  # Sub-system #2 for updating orientation
  ############################################################################
  [euler_rodrigues_2]
    type = RotationMatrix
    from = 'state/orientation'
    to = 'state/orientation_matrix'
  []
  [elasticity_2]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.25'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'forces/tmp/elastic_strain'
    stress = 'state/internal/cauchy_stress'
  []
  [orientation_rate]
    type = OrientationRate
    elastic_strain = 'forces/tmp/elastic_strain'
  []
  [plastic_spin]
    type = PlasticVorticity
  []
  [slip_strength_2]
    type = SingleSlipStrengthMap
    constant_strength = 50.0
    slip_hardening = 'forces/tmp/internal/slip_hardening'
  []
  [integrate_orientation]
    type = WR2ImplicitExponentialTimeIntegration
    variable = 'state/orientation'
  []
  [implicit_rate_2]
    type = ComposedModel
    models = "euler_rodrigues_2 elasticity_2 resolved_shear
              plastic_deformation_rate plastic_spin
              slip_rule slip_strength_2 orientation_rate
              integrate_orientation"
  []
  [subsystem2]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate_2'
    solver = 'newton'
  []

  ############################################################################
  # Cache information from sub-system #1
  ############################################################################
  [cache_elastic_strain]
    type = CopySR2
    from = 'state/elastic_strain'
    to = 'forces/tmp/elastic_strain'
  []
  [cache_slip_hardening]
    type = CopyScalar
    from = 'state/internal/slip_hardening'
    to = 'forces/tmp/internal/slip_hardening'
  []
  [cache1]
    type = ComposedModel
    models = 'cache_elastic_strain cache_slip_hardening'
  []

  ############################################################################
  # Cache information from sub-system #2
  ############################################################################
  [cache2]
    type = CopyRot
    from = 'state/orientation'
    to = 'forces/tmp/orientation'
  []

  ############################################################################
  # Sequentially update sub-system #1 and sub-system #2
  ############################################################################
  [model]
    type = ComposedModel
    models = 'cache2 subsystem1 cache1 subsystem2'
    priority = 'cache2 subsystem1 cache1 subsystem2'
    additional_outputs = 'state/elastic_strain state/internal/slip_hardening'
  []
[]
