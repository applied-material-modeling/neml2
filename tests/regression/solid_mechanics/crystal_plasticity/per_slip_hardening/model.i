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

  [initial_dislocation_density]
    type = FullScalar
    batch_shape = '(20,12)'
    value = 1.0e1
    intermediate_dimension = 1
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'deformation_rate'
    force_SR2_values = 'deformation_rate'
    force_WR2_names = 'vorticity'
    force_WR2_values = 'vorticity'
    ic_Scalar_names = 'dislocation_density'
    ic_Scalar_values = 'initial_dislocation_density'
    ic_Rot_names = 'orientation'
    ic_Rot_values = 'initial_orientation'
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
  [euler_rodrigues]
    type = RotationMatrix
    from = 'orientation'
    to = 'orientation_matrix'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.25'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
    stress = 'cauchy_stress'
  []
  [resolved_shear]
    type = ResolvedShear
    stress = 'cauchy_stress'
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
  [slip_rule]
    type = PowerLawSlipRule
    n = 8.0
    gamma0 = 2.0e-1
  []
  [slip_strength]
    type = DislocationObstacleStrengthMap
    dislocation_density = 'dislocation_density'
    alpha = 0.3
    mu = 1.0e5
    b = 1.0e-4
    constant_strength = 50.0
  []
  [dislocation_density_rate]
    type = PerSlipForestDislocationEvolution
    dislocation_density = 'dislocation_density'
    k1 = 1e2
    k2 = 40.0
  []
  [integrate_dislocation_density]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'dislocation_density'
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
    models = "euler_rodrigues elasticity orientation_rate resolved_shear
              elastic_stretch plastic_deformation_rate plastic_spin
              slip_rule slip_strength dislocation_density_rate
              integrate_dislocation_density integrate_elastic_strain integrate_orientation"
  []
[]

[EquationSystems]
  [es]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'elastic_strain orientation dislocation_density'
    residuals = 'elastic_strain_residual orientation_residual dislocation_density_residual'
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
  [cp_warmup_1]
    type = CrystalPlasticityStrainPredictor
    scale = 0.1
  []
  [cp_warmup_2]
    type = ConstantExtrapolationPredictor
    unknowns_Rot = 'orientation'
    unknowns_Scalar = 'dislocation_density'
  []
  [predictor]
    type = ComposedModel
    models = 'cp_warmup_1 cp_warmup_2'
  []
  [update]
    type = ImplicitUpdate
    equation_system = 'es'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model]
    type = ComposedModel
    models = 'update elasticity'
    additional_outputs = 'elastic_strain'
  []
[]
