# neml2
# Mixed-control polycrystal benchmark: ``nbatch`` random crystals share a
# single global mixed-control state (one deformation-rate component
# prescribed, the rest stress-controlled). Unlike every other benchmark in
# this directory, ``nbatch`` here scales the *sub-batch* axis (number of
# grains) rather than the dynamic-batch axis. The equation system has two
# unknown groups: group 0 is the per-crystal unknowns (elastic_strain,
# orientation, slip_hardening) with sub_batch=(nbatch,); group 1 is the
# global mixed-control unknowns (deformation_rate, target_cauchy_stress)
# with sub_batch=(). With per-(residual, unknown) seeds in the chain
# rule, the per-grain Jacobian block lands in compact (B, N, k_p, k_p)
# block-diagonal storage automatically -- no IStructure flag needed.
#
# Solver: SchurComplement eliminates group 0 first via a per-block DenseLU
# (each block solves one grain's residual with the global unknowns held
# frozen), then DenseLU on the small (1-grain-equivalent) global Schur
# residual. If the Schur factorisation is doing its job, wall time should
# scale ~linearly in nbatch (per-grain block solves dominate, all
# parallelised across the sub-batch). Sub-linear plateau would mean we're
# below the GPU/cache occupancy knee; super-linear growth would mean Schur
# is failing to localise the cost.

[Settings]
  # Per-variable AOTI example shapes. The per-crystal IC inputs declare
  # sub_batch=(${nbatch},) with the ``:grain`` label so the per-axis
  # preservation path activates and ``A_pp`` lands in block-diagonal
  # ``(B, nbatch, k_p, k_p)`` storage instead of fully dense
  # ``(B, nbatch*k_p, nbatch*k_p)``. Without ``:grain`` the SchurComplement
  # block structure still gives some parallelism but each per-Newton-iter
  # LU costs O((nbatch*k_p)^3) instead of O(nbatch*k_p^3) -- infeasible at
  # large nbatch. Without sub_batch declared at all,
  # ``system.initialize()`` falls back to ``((2,), ())`` for every input
  # and the chain rule traces a shape-mismatched Jacobian.
  [example_batch_shape]
    elastic_strain~1 = '(2; ${nbatch})'
    orientation~1 = '(2; ${nbatch})'
    slip_hardening~1 = '(2; ${nbatch})'
    deformation_rate~1 = '(2,)'
    target_cauchy_stress~1 = '(2,)'
    control = '(2,)'
    prescribed = '(2,)'
    vorticity = '(2,)'
    t = '(2,)'
    t~1 = '(2,)'
  []
[]

[Tensors]
  # HIT-substituted shim so the verbatim triple-quoted Python blocks below
  # can reference nbatch as a bare identifier. ${...} substitution only
  # works inside single-line single-quoted HIT strings; triple-quoted
  # blocks are passed verbatim to the Python eval namespace.
  [nbatch]
    type = Python
    expr = '${nbatch}'
  []

  # 100-step load history -- matches the regression test the model.i was
  # derived from. dynamic_batch dim stays at 1 since the timesteps are
  # the only outer batch loop; nbatch lives on the sub-batch axis.
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

  # Per-crystal ICs on the sub-batch axis. Random orientations seeded for
  # reproducibility across benchmark runs (the actual values don't change
  # the Schur cost model -- only the per-block solver work scales with
  # ${nbatch}). At nbatch=5 the seed-0 draw produces a representative
  # spread of orientations comparable to the legacy hand-picked set.
  [initial_orientation]
    type = Python
    expr = 'Rot(torch.randn(${nbatch}, 3, dtype=torch.float64), sub_batch_ndim=1)'
  []
  [initial_elastic_strain]
    type = Python
    expr = 'SR2(torch.zeros(${nbatch}, 6, dtype=torch.float64), sub_batch_ndim=1)'
  []
  [initial_slip_hardening]
    type = Python
    expr = 'Scalar(torch.zeros(${nbatch}, dtype=torch.float64), sub_batch_ndim=1)'
  []

  # Prescribed forces -- global per timestep, no sub-batch. ``control``
  # masks the deformation-rate (1) vs stress (0) components; here only
  # the first axial component is deformation-rate-prescribed at 1e-3,
  # everything else is stress-free.
  [control]
    type = Python
    expr = 'SR2.fill(1.0, 0, 0, 0, 0, 0).dynamic_batch.expand(100, 1)'
  []
  [prescribed]
    type = Python
    expr = 'SR2.fill(1e-3, 0, 0, 0, 0, 0).dynamic_batch.expand(100, 1)'
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

    prescribed_SR2_names = 'prescribed control'
    prescribed_SR2_values = 'prescribed control'
    prescribed_WR2_names = 'vorticity'
    prescribed_WR2_values = 'vorticity'

    ic_Rot_names = 'orientation'
    ic_Rot_values = 'initial_orientation'
    ic_SR2_names = 'elastic_strain'
    ic_SR2_values = 'initial_elastic_strain'
    ic_Scalar_names = 'slip_hardening'
    ic_Scalar_values = 'initial_slip_hardening'
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
    reduces = 'grain'
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
    structure = 'block dense'
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
