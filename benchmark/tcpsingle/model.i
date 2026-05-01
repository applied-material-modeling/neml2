[Tensors]
  [end_time]
    type = FullScalar
    value = 5000
    batch_shape = '(${nbatch})'
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
    value = 0.0
  []
  [dyy]
    type = FullScalar
    batch_shape = '(${nbatch})'
    value = 0.0001
  []
  [dzz]
    type = FullScalar
    batch_shape = '(${nbatch})'
    value = -0.0001
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
    value = 0.0
  []
  [w2]
    type = FullScalar
    batch_shape = '(${nbatch})'
    value = 0.0
  []
  [w3]
    type = FullScalar
    batch_shape = '(${nbatch})'
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
    type = MillerIndex
    values = '1 1 0'
  []
  [splanes]
    type = MillerIndex
    values = '1 1 1'
  []
  [initial_orientation]
    type = Orientation
    quantity = ${nbatch}
    values = '30 60 45'
    normalize = true
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model_with_stress'
    prescribed_time = 'times'
    force_SR2_names = 'deformation_rate'
    force_SR2_values = 'deformation_rate'
    force_WR2_names = 'vorticity'
    force_WR2_values = 'vorticity'
    ic_Rot_names = 'orientation'
    ic_Rot_values = 'initial_orientation'
    device = ${device}
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
    from = 'orientation'
    to = 'orientation_matrix'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
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
    n = 6.0
    gamma0 = 1.0
  []
  [slip_strength]
    type = SingleSlipStrengthMap
    constant_strength = 30.0
  []
  [voce_hardening]
    type = VoceSingleSlipHardeningRule
    initial_slope = 500.0
    saturated_hardening = 10.0
  []
  [integrate_slip_hardening]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'slip_hardening'
  []
  [integrate_elastic_strain]
    type = SR2BackwardEulerTimeIntegration
    variable = 'elastic_strain'
  []
  [integrate_orientation]
    type = WR2ImplicitExponentialTimeIntegration
    variable = 'orientation'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'euler_rodrigues elasticity orientation_rate resolved_shear elastic_stretch plastic_deformation_rate plastic_spin sum_slip_rates slip_rule slip_strength voce_hardening integrate_slip_hardening integrate_elastic_strain integrate_orientation'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'elastic_strain slip_hardening orientation'
  []
[]

[Solvers]
  [newton]
    type = NewtonWithLineSearch
    max_its = 500
    linesearch_cutback = 2.0
    linesearch_stopping_criteria = 1.0e-3
    max_linesearch_iterations = 5
    rel_tol = 1e-4
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [cp_warmup_1]
    type = CrystalPlasticityStrainPredictor
    scale = 0.05
    threshold = 1e-6
  []
  [cp_warmup_2]
    type = ConstantExtrapolationPredictor
    unknowns_Rot = 'orientation'
    unknowns_Scalar = 'slip_hardening'
  []
  [predictor]
    type = ComposedModel
    models = 'cp_warmup_1 cp_warmup_2'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model_with_stress]
    type = ComposedModel
    models = 'model elasticity'
    additional_outputs = 'elastic_strain'
  []
[]
