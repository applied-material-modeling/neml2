nstep = 100
nbatch = 1
ncrystal = 5





[Tensors]
  [times]
    type = LinspaceScalar
    start = 0
    end = 50
    nstep = ${nstep}
    shape_manipulations = 'dynamic_unsqueeze dynamic_expand'
    shape_manipulation_args = '(-1) (${nstep},${nbatch})'
  []

  [sdirs]
    type = MillerIndex
    values = '1 1 0'
  []
  [splanes]
    type = MillerIndex
    values = '1 1 1'
  []

  # Initial conditions for the state variables
  # Normally we would fill zeros if an initial state is not specified
  # Here, however, we specify the initial elastic strain and slip hardening
  # so that they have the correct shape with intermediate dimensions
  [initial_orientation]
    type = Rot
    values = "-0.269981 -0.299844 -0.86408
              0.209546 0.192014 0.514051
              -0.0251234 -0.0175916 -0.636644
              -0.146257 -0.0475218 -0.970804
              -0.174458 -0.302169 -0.523373"
    batch_shape = (${ncrystal})
    intermediate_dimension = 1
  []
  [initial_elastic_strain]
    type = FillSR2
    values = '0'
    shape_manipulations = 'intmd_expand'
    shape_manipulation_args = '(${ncrystal})'
  []
  [initial_slip_hardening]
    type = Scalar
    values = '0'
    shape_manipulations = 'intmd_expand'
    shape_manipulation_args = '(${ncrystal})'
  []

  # For mixed control:
  # Control signals above 0.5 -> strain control (via deformation rate)
  #                 below 0.5 -> stress control (via mean cauchy stress)
  # Here we want to model uniaxial tension, so only the first component (xx) is 1, rest are 0
  [control]
    type = FillSR2
    values = '1 0 0 0 0 0'
    shape_manipulations = 'dynamic_expand'
    shape_manipulation_args = '(${nstep},${nbatch})'
  []
  [prescribed]
    type = FillSR2
    values = '1e-3 0 0 0 0 0' # 1e-3 deformation rate, rest are zero mean cauchy stress
    shape_manipulations = 'dynamic_expand'
    shape_manipulation_args = '(${nstep},${nbatch})'
  []

  # The solution is unique up to some spin
  # Here, we fix it to be zero for simplicity
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
    model = 'model'
    prescribed_time = 'times'

    force_SR2_names = 'forces/prescribed forces/control'
    force_SR2_values = 'prescribed control'
    force_WR2_names = 'forces/vorticity'
    force_WR2_values = 'vorticity'

    ic_Rot_names = 'state/orientation'
    ic_Rot_values = 'initial_orientation'
    ic_SR2_names = 'state/elastic_strain'
    ic_SR2_values = 'initial_elastic_strain'
    ic_Scalar_names = 'state/internal/slip_hardening'
    ic_Scalar_values = 'initial_slip_hardening'

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
    lattice_parameter = 1
    slip_directions = 'sdirs'
    slip_planes = 'splanes'
  []
[]

[Models]
  ############################################################################
  # Mixed control
  #
  # Note this is _global_ -- there is no per-crystal mixed control here
  # The shapes of state/deformation_rate and state/target_cauchy_stress are
  # (...; ; 6), i.e., no intermediate dimension for crystals
  ############################################################################
  [mixed_control]
    type = MixedControlSetup
    control = 'forces/control'
    mixed_state = 'state/mixed_state'
    fixed_values = 'forces/prescribed'
    above_variable = 'state/deformation_rate'
    below_variable = 'state/target_cauchy_stress'
  []
  ############################################################################
  # Per-crystal update
  #
  # This is the beef of the update -- all models here operate on
  # per-crystal variables with intermediate dimension. We are solving for
  # three variables per crystal:
  #   state/elastic_strain           (...; ncrystal; 6)
  #   state/orientation              (...; ncrystal; 3)
  #   state/internal/slip_hardening  (...; ncrystal;)
  #
  # Their corresponding residuals are stored in:
  #   residual/elastic_strain           (...; ncrystal; 6)
  #   residual/orientation              (...; ncrystal; 3)
  #   residual/internal/slip_hardening  (...; ncrystal;)
  #
  # For each crystal, the Jacobian looks like:
  #
  #        | dr_e/de  dr_e/dq  dr_e/dh |
  #   J =  | dr_q/de  dr_q/dq  dr_q/dh |
  #        | dr_h/de  dr_h/dq  dr_h/dh |
  #
  # Here, e = elastic_strain, q = orientation, h = slip_hardening
  # J is generally a dense matrix. And remember this Jacobian is per-crystal,
  # and there's no coupling between crystals other than through the global
  # mixed control constraint -- to be discussed later.
  ############################################################################
  [euler_rodrigues]
    type = RotationMatrix
    from = 'state/orientation'
    to = 'state/orientation_matrix'
  []
  [elastic_tensor]
    type = CubicElasticityTensor
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO SHEAR_MODULUS'
    coefficients = '209016 0.307 60355.0'
  []
  [elasticity]
    type = GeneralElasticity
    elastic_stiffness_tensor = 'elastic_tensor'
    strain = 'state/elastic_strain'
    stress = 'state/cauchy_stress'
  []
  [resolved_shear]
    type = ResolvedShear
    stress = 'state/cauchy_stress'
  []
  [elastic_stretch]
    type = ElasticStrainRate
    deformation_rate = 'state/deformation_rate'
    elastic_strain_rate = 'state/elastic_strain_rate'
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
  [per_crystal_update]
    type = ComposedModel
    models = "elasticity euler_rodrigues
              orientation_rate resolved_shear
              elastic_stretch
              plastic_deformation_rate plastic_spin
              sum_slip_rates slip_rule slip_strength voce_hardening
              integrate_slip_hardening
              integrate_elastic_strain
              integrate_orientation"
    # We need to also output cauchy stress as the Taylor approximation
    # applies constraint on mean cauchy stress
    additional_outputs = 'state/cauchy_stress'
  []
  ############################################################################
  # Global constraint
  #
  # Now that we have a per-crystal update model, mapping from
  #   state/deformation_rate
  #   state/elastic_strain
  #   state/orientation
  #   state/internal/slip_hardening
  # To
  #   residual/elastic_strain
  #   residual/orientation
  #   residual/internal/slip_hardening
  #
  # We need to enforce the global mixed control constraint:
  #   r(m) = target(σ) - mean(σ), for stress-controlled components
  #          0,                   for strain-controlled components
  #
  # which enforces that the mean cauchy stress across all crystals meets the
  # target mean cauchy stress for the components under stress control.
  ############################################################################
  [mean_stress]
    type = SR2IntermediateMean
    from = 'state/cauchy_stress'
    to = 'state/mean_cauchy_stress'
  []
  [match_mean_cauchy_stress]
    type = SR2LinearCombination
    from_var = 'state/target_cauchy_stress state/mean_cauchy_stress'
    to_var = 'residual/mean_cauchy_stress'
    coefficients = '1 -1'
  []
  [mixed_control_constraint]
    type = MixedControlConstraint
    control = 'forces/control'
    below_residual = 'residual/mean_cauchy_stress'
    mixed_residual = 'residual/mixed_state'
  []
  [global_constraint]
    type = ComposedModel
    models = 'mean_stress match_mean_cauchy_stress mixed_control_constraint'
  []
  ############################################################################
  # Putting everything together
  #
  # The composition itself is pretty straightforward. To summarize:
  #   1) mixed_control maps from
  #        - forces/control
  #        - forces/prescribed
  #        - state/mixed_state
  #      to
  #        - state/deformation_rate
  #        - state/target_cauchy_stress
  #   2) per_crystal_update maps from
  #        - state/deformation_rate
  #        - state/elastic_strain
  #        - state/orientation
  #        - state/internal/slip_hardening
  #      to
  #        - residual/elastic_strain
  #        - residual/orientation
  #        - residual/internal/slip_hardening
  #        - state/cauchy_stress
  #   3) global_constraint maps from
  #        - forces/control
  #        - state/target_cauchy_stress
  #        - state/cauchy_stress
  #      to
  #        - residual/mixed_state
  #
  # Composing them all together gives us the full nonlinear system which maps
  # from
  #   - forces/control                 (...; ; 6)
  #   - forces/prescribed              (...; ; 6)
  #   ----------------------------------------------------
  #   - state/mixed_state              (...; ; 6)
  #   - state/elastic_strain           (...; ncrystal; 6)
  #   - state/orientation              (...; ncrystal; 3)
  #   - state/internal/slip_hardening  (...; ncrystal;)
  # to
  #   - residual/mixed_state              (...; ; 6)
  #   - residual/elastic_strain           (...; ncrystal; 6)
  #   - residual/orientation              (...; ncrystal; 3)
  #   - residual/internal/slip_hardening  (...; ncrystal;)
  #
  # This is a square system! Each state variable (unknown) maps to a residual
  # of the same size.
  #
  # It is easy to see that the system has (6 + ncrystal * (6 + 3 + 1))
  # degrees of freedom. Naively, one may treat the entire system as a big,
  # dense system. However, that would scale extremely inefficiently as the number
  # of crystals increases. Both memory and complexity grows like O(ncrystal^3).
  #
  # Fortunately, we can exploit the structure of the system to solve it
  # efficiently. Note that the only coupling between crystals is through
  # the global mixed control constraint, which only adds (6) equations to
  # the entire system. This means that if we ignore the global constraint
  # for a moment, the Jacobian of the remaining system is block-diagonal.
  #
  # For a system like this, we can use a Schur complement approach to
  # efficiently compute the Newton update. The complexity of this approach
  # is O(ncrystal) when ncrystal >> ndof, as opposed to O(ncrystal^3) for a
  # dense solve. That is a big win! Memory wise, since we are not storing the
  # off-diagonal zeros, we also save a lot of memory.
  ############################################################################
  [implicit_model]
    type = ComposedModel
    models = 'mixed_control per_crystal_update global_constraint'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_model'
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
    schur_variables = 'state/mixed_state'
    schur_linear_solver = 'lu'
    primary_linear_solver = 'lu'
  []
[]

[Models]
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
[]
