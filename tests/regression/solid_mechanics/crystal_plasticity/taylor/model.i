nstep = 100
nbatch = 5
ncrystal = 4

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
    nstep = ${nstep}
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
    shape_manipulations = 'intmd_expand'
    shape_manipulation_args = '(${ncrystal})'
  []

  # For mixed control:
  # Control signals above 0.5 -> strain control (via deformation rate)
  #                 below 0.5 -> stress control
  # Here we want to model uniaxial tension, so only the first component is 1, rest are 0
  [control]
    type = FillSR2
    values = '1 0 0 0 0 0'
    shape_manipulations = 'dynamic_expand'
    shape_manipulation_args = '(${nstep},${nbatch})'
  []
  [prescribed]
    type = FillSR2
    values = '0.1 0 0 0 0 0' # 0.1 deformation rate, 0 stress
    shape_manipulations = 'dynamic_expand'
    shape_manipulation_args = '(${nstep},${nbatch})'
  []

  # The solution is unique up to some spin -- we fix it to be zero for simplicity
  [vorticity]
    type = FillWR2
    values = '0 0 0'
    shape_manipulations = 'dynamic_expand'
    shape_manipulation_args = '(${nstep},${nbatch})'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model_with_stress'
    prescribed_time = 'times'
    force_SR2_names = 'forces/prescribed forces/control'
    force_SR2_values = 'prescribed control'
    force_WR2_names = 'forces/vorticity'
    force_WR2_values = 'vorticity'
    ic_Rot_names = 'state/orientation'
    ic_Rot_values = 'initial_orientation'
    var_with_intmd_dims = 'state/internal/slip_hardening'
    var_intmd_shapes = '(${ncrystal})'
    predictor = 'PREVIOUS_STATE'
    save_as = 'result.pt'
    verbose = true
    show_model_info = true
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
    verbose = true
  []
[]

[Data]
  [crystal_geometry]
    type = CubicCrystal
    lattice_parameter = 1
    slip_directions = 'sdirs'
    slip_planes = 'splanes'
  []
[]

[Models]
  [mixed_control]
    type = MixedControlSetup
    mixed_state = 'state/mixed_state'
    fixed_values = 'forces/prescribed'
    above_variable = 'state/deformation_rate'
    below_variable = 'state/cauchy_stress'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.25'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'state/elastic_strain'
    stress = 'state/cauchy_stress'
    compliance = true
  []
  [euler_rodrigues]
    type = RotationMatrix
    from = 'state/orientation'
    to = 'state/orientation_matrix'
  []
  [resolved_shear]
    type = ResolvedShear
    stress = 'state/cauchy_stress'
  []
  [elastic_stretch]
    type = ElasticStrainRate
    deformation_rate = 'state/deformation_rate'
    elastic_strain_rate = 'state/crystal_elastic_strain_rate'
  []
  [elastic_stretch_avg]
    type = SR2IntermediateMean
    from = 'state/crystal_elastic_strain_rate'
    to = 'state/elastic_strain_rate'
    dim = -1
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
  [rename_residual]
    type = CopySR2
    from = 'residual/elastic_strain'
    to = 'residual/mixed_state'
  []
  [integrate_orientation]
    type = WR2ImplicitExponentialTimeIntegration
    variable = 'state/orientation'
  []

  [implicit_rate]
    type = ComposedModel
    models = "mixed_control elasticity euler_rodrigues
              orientation_rate resolved_shear
              elastic_stretch elastic_stretch_avg
              plastic_deformation_rate plastic_spin
              sum_slip_rates slip_rule slip_strength voce_hardening
              integrate_slip_hardening integrate_elastic_strain rename_residual integrate_orientation"
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
  [model_with_stress]
    type = ComposedModel
    models = 'model mixed_control elasticity'
    additional_outputs = 'state/cauchy_stress'
  []
[]
