# neml2
# nlprecond testbed case: cp_decoupled_variational
# Derived from studies/nlprecond/cases/cp_decoupled/model.i. Sub-system #1 is
# made EXACTLY VARIATIONAL:
#   * slip rates carried as unknowns, closed by the inverted residual
#   * slip strength evaluated from the PREVIOUS step's hardening (lagged tauc)
#   * orientation is already lagged here (sub-system #1 uses orientation~1)
# With tauc and the orientation both frozen during the solve, sub-system #1's
# residual is exactly dI/dgdot for
#     I = Psi(Ee_trial - dt*sum gdot_i M_i) + dt*sum phi(gdot_i; tauc_n)
# and nothing else is an unknown, so I is a genuine merit function. The coupled
# case cannot test this: orientation is an unknown there, contributing to the
# residual but not to I, so dI = 0 does not imply R = 0.
#
# Originally from tests/regression/solid_mechanics/crystal_plasticity/single_crystal_decoupled/model.i
#
# Differences from the parent regression scenario:
#   * [Drivers/regression] removed (no gold reference; this is a solver study)
#   * knobs exposed as HIT ${...} substitutions, supplied by studies.nlprecond.harness:
#       nbatch    -- dynamic batch members
#       npoint    -- time points; the driver takes npoint-1 steps
#       tfrac     -- fraction of the parent's full load history to cover. The
#                    harness sets it so the PER-STEP INCREMENT stays fixed as
#                    npoint shrinks -- fewer steps, not bigger ones. Increment
#                    size is a separate knob (dt_scale), which scales tfrac.
#       flow_n    -- PowerLawSlipRule rate-sensitivity exponent n
#       ls_iters  -- max_linesearch_iterations; 1 == full Newton step (no line search)
#       max_its   -- Newton iteration cap
#   * [Solvers/newton] is always NewtonWithLineSearch so ls_iters spans both arms
#   * the predictor wiring is left intact; the harness strips it for the
#     "nopred" arms by deleting the `predictor = '...'` line(s)
[Tensors]
  # end_time = LinspaceScalar(1, 10, nbatch) -> shape (nbatch,)
  [end_time]
    type = Python
    expr = 'linspace(Scalar(1.0).dynamic_batch, Scalar(10.0).dynamic_batch, ${nbatch})'
  []
  # times = LinspaceScalar(0, end_time, npoint) -> shape (npoint, nbatch)
  [times]
    type = Python
    expr = 'Scalar(end_time.data.unsqueeze(0) * torch.linspace(0.0, ${tfrac}, ${npoint}, dtype=torch.float64).unsqueeze(-1))'
  []
  # deformation_rate single = FillSR2(0.1, -0.05, -0.05) batched (nbatch,)
  [deformation_rate_single]
    type = Python
    expr = 'SR2.fill(0.1, -0.05, -0.05, 0.0, 0.0, 0.0).dynamic_batch.expand(${nbatch})'
  []
  # deformation_rate = LinspaceSR2(d_single, d_single, npoint) -> shape (npoint, nbatch, 6)
  [deformation_rate]
    type = Python
    expr = 'SR2(deformation_rate_single.data.unsqueeze(0).expand(${npoint}, ${nbatch}, 6).contiguous())'
  []
  # vorticity single = FillWR2(0.1, -0.05, -0.05) batched (nbatch,)
  [vorticity_single]
    type = Python
    expr = 'WR2(torch.tensor([0.1, -0.05, -0.05], dtype=torch.float64).unsqueeze(0).expand(${nbatch}, 3).contiguous())'
  []
  # vorticity = LinspaceWR2(w_single, w_single, npoint) -> shape (npoint, nbatch, 3)
  [vorticity]
    type = Python
    expr = 'WR2(vorticity_single.data.unsqueeze(0).expand(${npoint}, ${nbatch}, 3).contiguous())'
  []

  [slip_rate_ic]
    type = Python
    expr = 'Scalar(torch.full((${nbatch}, 12), ${gdot_seed}, dtype=torch.float64), sub_batch_ndim=1)'
  []

  # Crystal geometry inputs: lattice parameter + slip direction + slip plane
  [a]
    type = Python
    expr = 'Scalar(1.0)'
  []
  [sdirs]
    type = Python
    expr = 'MillerIndex(torch.tensor([1, 1, 0], dtype=torch.int64))'
  []
  [splanes]
    type = Python
    expr = 'MillerIndex(torch.tensor([1, 1, 1], dtype=torch.int64))'
  []

  # Initial orientation = FillRot(R1, R2, R3, method='standard'):
  # convert standard Rodrigues r_std to modified-Rodrigues parameters via
  # r = r_std / (sqrt(|r_std|^2 + 1) + 1). Shape (nbatch, 3).
  [initial_orientation]
    type = Python
    expr = 'MRP((lambda r: r / (torch.sqrt((r * r).sum(-1, keepdim=True) + 1.0) + 1.0))(torch.stack([torch.linspace(0.0, 0.75, ${nbatch}, dtype=torch.float64), torch.linspace(0.0, -0.25, ${nbatch}, dtype=torch.float64), torch.linspace(-0.1, 0.1, ${nbatch}, dtype=torch.float64)], dim=-1)))'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_SR2_names = 'deformation_rate'
    prescribed_SR2_values = 'deformation_rate'
    prescribed_WR2_names = 'vorticity'
    prescribed_WR2_values = 'vorticity'
    ic_Scalar_names = 'slip_rates'
    ic_Scalar_values = 'slip_rate_ic'
    ic_MRP_names = 'orientation'
    ic_MRP_values = 'initial_orientation'
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
    from = 'orientation~1'
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
  [plastic_deformation_rate]
    type = PlasticDeformationRate
  []
  [sum_slip_rates]
    type = SumSlipRates
  []
  [slip_rule]
    type = PowerLawSlipRuleResidual
    n = '${flow_n}'
    gamma0 = 2.0e-1
    cutoff = '${gdot_cutoff}'
  []
  [slip_strength]
    type = SingleSlipStrengthMap
    constant_strength = 50.0
    slip_hardening = 'slip_hardening~1'
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
  [implicit_rate_1]
    type = ComposedModel
    models = 'euler_rodrigues_1 elasticity resolved_shear
              elastic_stretch plastic_deformation_rate
              sum_slip_rates slip_rule slip_strength voce_hardening
              integrate_slip_hardening integrate_elastic_strain'
  []

  ############################################################################
  # Sub-system #2 for updating orientation
  ############################################################################
  [euler_rodrigues_2]
    type = RotationMatrix
    from = 'orientation'
    to = 'orientation_matrix'
  []
  [orientation_rate]
    type = OrientationRate
    elastic_strain = 'elastic_strain'
  []
  [plastic_spin]
    type = PlasticVorticity
  []
  [integrate_orientation]
    type = WR2ImplicitExponentialTimeIntegration
    variable = 'orientation'
  []
  [implicit_rate_2]
    type = ComposedModel
    models = 'euler_rodrigues_2 elasticity resolved_shear
              plastic_deformation_rate plastic_spin
              orientation_rate
              integrate_orientation'
  []
[]

[EquationSystems]
  [eq_sys_1]
    type = NonlinearSystem
    model = 'implicit_rate_1'
    unknowns = 'slip_rates; elastic_strain slip_hardening'
    residuals = 'slip_rates_residual; elastic_strain_residual slip_hardening_residual'
    structure = 'block dense'
  []
  [eq_sys_2]
    type = NonlinearSystem
    model = 'implicit_rate_2'
    unknowns = 'orientation'
    residuals = 'orientation_residual'
  []
[]

[Solvers]
  [newton]
    type = NewtonWithLineSearch
    max_linesearch_iterations = '${ls_iters}'
    max_its = '${max_its}'
    linear_solver = 'schur'
  []
  [lu]
    type = DenseLU
  []
  [newton_dense]
    type = NewtonWithLineSearch
    max_linesearch_iterations = '${ls_iters}'
    max_its = '${max_its}'
    linear_solver = 'lu'
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
  ############################################################################
  # Update sub-system #1
  ############################################################################
  [cp_warmup_1]
    type = CrystalPlasticityStrainPredictor
    scale = 0.1
  []
  [cp_warmup_2]
    type = ConstantExtrapolationPredictor
    unknowns_Scalar = 'slip_hardening slip_rates'
  []
  [predictor1]
    type = ComposedModel
    models = 'cp_warmup_1 cp_warmup_2'
  []
  [subsystem1]
    type = ImplicitUpdate
    equation_system = 'eq_sys_1'
    solver = 'newton'
    predictor = 'predictor1'
  []
  ############################################################################
  # Update sub-system #2
  ############################################################################
  [predictor2]
    type = ConstantExtrapolationPredictor
    unknowns_MRP = 'orientation'
  []
  [subsystem2]
    type = ImplicitUpdate
    equation_system = 'eq_sys_2'
    solver = 'newton_dense'
    predictor = 'predictor2'
  []
  ############################################################################
  # Sequentially update sub-system #1 and sub-system #2
  ############################################################################
  [model]
    type = ComposedModel
    models = 'subsystem1 subsystem2'
    additional_outputs = 'elastic_strain slip_hardening slip_rates'
  []
[]
