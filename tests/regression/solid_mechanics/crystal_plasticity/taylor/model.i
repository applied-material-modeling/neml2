# neml2
# Taylor polycrystal aggregate: 5 crystals share a global stress / deformation
# rate. Exercises the BLOCK + DENSE Schur factorisation — group 0 is the
# per-crystal unknowns (elastic_strain, orientation, slip_hardening) with
# uniform sub_batch=(5,), group 1 is the global mixed-control unknowns
# (deformation_rate, target_cauchy_stress) with sub_batch=().
[Tensors]
  [times]
    type = Python
    expr = 'Scalar(torch.linspace(0.0, 50.0, 100, dtype=torch.float64).reshape(100, 1))'
  []
  [sdirs]
    type = Python
    expr = 'MillerIndex(torch.tensor([1, 1, 0], dtype=torch.int64))'
  []
  [splanes]
    type = Python
    expr = 'MillerIndex(torch.tensor([1, 1, 1], dtype=torch.int64))'
  []
  [elastic_tensor]
    type = Python
    expr = 'SSR4(torch.tensor([[287111.0283, 127190.5998, 127190.5998, 0, 0, 0], [127190.5998, 287111.0283, 127190.5998, 0, 0, 0], [127190.5998, 127190.5998, 287111.0283, 0, 0, 0], [0, 0, 0, 120710, 0, 0], [0, 0, 0, 0, 120710, 0], [0, 0, 0, 0, 0, 120710]], dtype=torch.float64))'
  []
  # Per-crystal ICs: leading (5,) is the per-crystal sub-batch axis,
  # declared via sub_batch_ndim=1 so the equation system treats it as a
  # block-diagonal (BLOCK) axis rather than a dynamic-batch axis.
  [initial_orientation]
    type = Python
    expr = 'Rot(torch.tensor([[-0.269981, -0.299844, -0.86408], [0.209546, 0.192014, 0.514051], [-0.0251234, -0.0175916, -0.636644], [-0.146257, -0.0475218, -0.970804], [-0.174458, -0.302169, -0.523373]], dtype=torch.float64), sub_batch_ndim=1)'
  []
  [initial_elastic_strain]
    type = Python
    expr = 'SR2(torch.zeros(5, 6, dtype=torch.float64), sub_batch_ndim=1)'
  []
  [initial_slip_hardening]
    type = Python
    expr = 'Scalar(torch.zeros(5, dtype=torch.float64), sub_batch_ndim=1)'
  []
  # Prescribed forces (no sub-batch — they are global per timestep).
  [control]
    type = Python
    expr = 'SR2(torch.tensor([1.0, 0, 0, 0, 0, 0], dtype=torch.float64).reshape(1, 6).expand(100, 1, 6).contiguous())'
  []
  [prescribed]
    type = Python
    expr = 'SR2(torch.tensor([1e-3, 0, 0, 0, 0, 0], dtype=torch.float64).reshape(1, 6).expand(100, 1, 6).contiguous())'
  []
  [vorticity]
    type = Python
    expr = 'WR2(torch.zeros(100, 1, 3, dtype=torch.float64))'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'

    force_SR2_names = 'prescribed control'
    force_SR2_values = 'prescribed control'
    force_WR2_names = 'vorticity'
    force_WR2_values = 'vorticity'

    ic_Rot_names = 'orientation'
    ic_Rot_values = 'initial_orientation'
    ic_SR2_names = 'elastic_strain'
    ic_SR2_values = 'initial_elastic_strain'
    ic_Scalar_names = 'slip_hardening'
    ic_Scalar_values = 'initial_slip_hardening'

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
    lattice_parameter = 1
    slip_directions = 'sdirs'
    slip_planes = 'splanes'
  []
[]

[Models]
  ############################################################################
  # Mixed control (global)
  ############################################################################
  [mixed_control]
    type = MixedControlSetup
    x_above = 'deformation_rate'
    x_below = 'target_cauchy_stress'
    y = 'mixed_state'
  []
  [y_constraint]
    type = SR2LinearCombination
    from = 'mixed_state prescribed'
    to = 'y_residual'
    weights = '1 -1'
  []

  ############################################################################
  # Per-crystal update
  ############################################################################
  [euler_rodrigues]
    type = RotationMatrix
    from = 'orientation'
    to = 'orientation_matrix'
  []
  [elasticity]
    type = GeneralElasticity
    elastic_stiffness_tensor = 'elastic_tensor'
    strain = 'elastic_strain'
    stress = 'cauchy_stress'
  []
  [resolved_shear]
    type = ResolvedShear
    stress = 'cauchy_stress'
  []
  [elastic_stretch]
    type = ElasticStrainRate
    deformation_rate = 'deformation_rate'
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
  [per_crystal_update]
    type = ComposedModel
    models = 'elasticity euler_rodrigues
              orientation_rate resolved_shear
              elastic_stretch
              plastic_deformation_rate plastic_spin
              sum_slip_rates slip_rule slip_strength voce_hardening
              integrate_slip_hardening
              integrate_elastic_strain
              integrate_orientation'
    additional_outputs = 'cauchy_stress'
  []

  ############################################################################
  # Global constraint
  ############################################################################
  [mean_stress]
    type = SR2IntermediateMean
    from = 'cauchy_stress'
    to = 'mean_cauchy_stress'
  []
  [match_mean_cauchy_stress]
    type = SR2LinearCombination
    from = 'target_cauchy_stress mean_cauchy_stress'
    to = 'target_cauchy_stress_residual'
    weights = '1 -1'
  []
  [global_constraint]
    type = ComposedModel
    models = 'mean_stress match_mean_cauchy_stress'
  []

  ############################################################################
  # Full implicit model
  ############################################################################
  [implicit_model]
    type = ComposedModel
    models = 'mixed_control y_constraint per_crystal_update global_constraint'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_model'
    unknowns = 'elastic_strain orientation slip_hardening; deformation_rate target_cauchy_stress'
    residuals = 'elastic_strain_residual orientation_residual slip_hardening_residual; y_residual target_cauchy_stress_residual'
    istructure = 'BLOCK DENSE'
  []
[]

[Solvers]
  [newton]
    type = NewtonWithLineSearch
    max_linesearch_iterations = 5
    linear_solver = 'schur'
  []
  [lu]
    type = DenseLU
  []
  [schur]
    type = SchurComplement
    residual_primary_group = '0'
    unknown_primary_group = '0'
    primary_solver = 'lu'
    schur_solver = 'lu'
  []
[]

[Models]
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_SR2 = 'elastic_strain deformation_rate target_cauchy_stress'
    unknowns_Rot = 'orientation'
    unknowns_Scalar = 'slip_hardening'
  []
  [model_bare]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [compute_mixed_state]
    type = MixedControlSetup
    x_above = 'deformation_rate'
    x_below = 'target_cauchy_stress'
    y = 'mixed_state'
  []
  [model]
    type = ComposedModel
    models = 'model_bare compute_mixed_state'
    additional_outputs = 'mixed_state'
  []
[]
