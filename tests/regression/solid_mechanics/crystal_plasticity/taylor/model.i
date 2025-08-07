[Tensors]
  [end_time]
    type = FullScalar
    batch_shape = '(3)'
    value = 5.0
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = end_time
    nstep = 100
  []
  [dxx]
    type = FullScalar
    batch_shape = '(3)'
    value = 0.0001
  []
  [dyy]
    type = FullScalar
    batch_shape = '(3)'
    value = -0.00005
  []
  [dzz]
    type = FullScalar
    batch_shape = '(3)'
    value = -0.00005
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
    batch_shape = '(3)'
    value = 0.0
  []
  [w2]
    type = FullScalar
    batch_shape = '(3)'
    value = 0.0
  []
  [w3]
    type = FullScalar
    batch_shape = '(3)'
    value = 0.0
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
  [initial_orientation]
    type = Orientation
    input_type = "euler_angles"
    angle_convention = "kocks"
    angle_type = "degrees"
    values = '45 50 51
              75 50 10
              10 5  60
              17 18 19
              30 60 90'
    additional_batch_shape = '(3)'
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
    cp_warmup = true
    cp_warmup_elastic_scale = 0.1
    save_as = 'result.pt'
  []
[]

[Solvers]
  [newton]
    type = Newton
    verbose = true
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
  [euler_rodrigues]
    type = RotationMatrix
    from = 'state/orientation'
    to = 'state/orientation_matrix'
    jit = false
  []
  [elastic_tensor]
    type = CubicElasticityTensor
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO SHEAR_MODULUS'
    coefficients = '209016 0.307 60355.0'
    jit = false
  []
  [elasticity]
    type = GeneralElasticity
    elastic_stiffness_tensor = 'elastic_tensor'
    strain = 'state/elastic_strain'
    stress = 'state/internal/cauchy_stress'
    jit = false
  []
  [resolved_shear]
    type = ResolvedShear
    jit = false
  []
  [elastic_stretch]
    type = ElasticStrainRate
    jit = false
  []
  [plastic_spin]
    type = PlasticVorticity
    jit = false
  []
  [plastic_deformation_rate]
    type = PlasticDeformationRate
    jit = false
  []
  [orientation_rate]
    type = OrientationRate
    jit = false
  []
  [sum_slip_rates]
    type = SumSlipRates
    jit = false
  []
  [slip_rule]
    type = PowerLawSlipRule
    n = 8.0
    gamma0 = 2.0e-1
    jit = false
  []
  [slip_strength]
    type = SingleSlipStrengthMap
    constant_strength = 50.0
    jit = false
  []
  [voce_hardening]
    type = VoceSingleSlipHardeningRule
    initial_slope = 500.0
    saturated_hardening = 50.0
    jit = false
  []
  [integrate_slip_hardening]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/internal/slip_hardening'
    jit = false
  []
  [integrate_elastic_strain]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/elastic_strain'
    jit = false
  []
  [integrate_orientation]
    type = WR2ImplicitExponentialTimeIntegration
    variable = 'state/orientation'
    jit = false
  []
  [average_stress]
    type = SR2ImplicitListReduction
    batched_variable = 'state/internal/cauchy_stress'
    variable = 'state/mean_cauchy_stress'
    jit = false
    batched_variable_list_size = 5
  []

  [implicit_rate]
    type = ComposedModel
    models = "euler_rodrigues elasticity orientation_rate resolved_shear
              elastic_stretch plastic_deformation_rate plastic_spin
              sum_slip_rates slip_rule slip_strength voce_hardening
              integrate_slip_hardening integrate_elastic_strain integrate_orientation average_stress"
    jit = false
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
[]
