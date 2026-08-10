# neml2
# nlprecond testbed case: cp_coupled_variational
# Derived from cp_coupled_inverted, plus one change that makes the update
# EXACTLY VARIATIONAL: the slip strength is evaluated from the PREVIOUS step's
# hardening (slip_hardening~1) instead of the current one.
#
# Why: with tauc frozen during the solve, the (elastic_strain, slip_rates)
# residuals are exactly dI/dgdot for
#     I = Psi(Ee_trial - dt*sum gdot_i M_i) + dt*sum phi(gdot_i; tauc_n)
# so I is a true merit function. With tauc implicit, dphi/dtauc * dtauc/dgdot
# is missing from the residual, the converged root is NOT I's minimizer (~1.5%
# off measured), and a line search on I stalls outright.
#
# The cost is a physics change: hardening lags by one step, an O(dt) error.
# slip_hardening is still integrated implicitly, so the system becomes block
# triangular -- (Ee, gdot) no longer depend on it, it depends on them.
#
# NOT derived from cp_coupled directly -- see below.
#
# Originally derived from studies/nlprecond/cases/cp_coupled/model.i -- SAME PHYSICS,
# different residual FORM. The power-law slip rule is stated implicitly in
# inverted form (tauc*sgn(g)|g/gamma0|^(1/n) - tau = 0,
# PowerLawSlipRuleResidual) with the slip rates carried as unknowns, instead of
# explicitly (g = gamma0|tau/tauc|^(n-1)(tau/tauc), PowerLawSlipRule).
#
# The slip rates are per-slip-system, so this promotes a SUB-BATCHED quantity
# to an unknown: for cubic <110>{111} the system grows by 12 per grain and the
# layout becomes BLOCK. slip_rates therefore needs a shaped initial condition
# (12 is the slip-system count of the [Data] crystal_geometry below).
#
# Those 12 unknowns are nearly free, though: each slip rate's residual involves
# only its own rate, so the (slip_rates, slip_rates) Jacobian block is exactly
# diagonal (verified: off-diagonal is identically zero). The equation system is
# therefore split into two groups with slip_rates FIRST -- 'block dense' with
# the BLOCK group primary, the arrowhead orientation SchurComplement supports --
# so the block condenses out per-site and the Schur complement is the ordinary
# 10x10. Putting the dense group first instead does not converge.
#
# Originally from tests/regression/solid_mechanics/crystal_plasticity/single_crystal_coupled/model.i
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
  # deformation_rate single = FillSR2(dxx=0.1, dyy=-0.05, dzz=-0.05) batched (nbatch,)
  [deformation_rate_single]
    type = Python
    expr = 'SR2.fill(0.1, -0.05, -0.05, 0.0, 0.0, 0.0).dynamic_batch.expand(${nbatch})'
  []
  # deformation_rate = LinspaceSR2(d_single, d_single, npoint) -> shape (npoint, nbatch, 6)
  [deformation_rate]
    type = Python
    expr = 'SR2(deformation_rate_single.data.unsqueeze(0).expand(${npoint}, ${nbatch}, 6).contiguous())'
  []
  # vorticity single = FillWR2(w1=0.1, w2=-0.05, w3=-0.05) batched (nbatch,)
  [vorticity_single]
    type = Python
    expr = 'WR2(torch.tensor([0.1, -0.05, -0.05], dtype=torch.float64).unsqueeze(0).expand(${nbatch}, 3).contiguous())'
  []
  # vorticity = LinspaceWR2(w_single, w_single, npoint) -> shape (npoint, nbatch, 3)
  [vorticity]
    type = Python
    expr = 'WR2(vorticity_single.data.unsqueeze(0).expand(${npoint}, ${nbatch}, 3).contiguous())'
  []

  # Step-1 seed for the slip-rate unknowns, shaped (nbatch, 12) with
  # sub_batch_ndim=1. The shape is required (with slip_rates an unknown the
  # driver would otherwise default it to base shape, 12x too short per grain);
  # the MAGNITUDE is what dominates the step-1 iteration count. A zero seed sits
  # inside the regularization cutoff, where the odd cubic's slope is ~1e17, so
  # Newton multiplies the slip rate by only ~100 per iteration and burns ~12
  # iterations climbing 20 decades to reach its physical value.
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
    model = 'model_with_stress'
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
    # THE variational change: lag the hardening that sets the slip strength.
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
  [integrate_orientation]
    type = WR2ImplicitExponentialTimeIntegration
    variable = 'orientation'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'euler_rodrigues elasticity orientation_rate resolved_shear
              elastic_stretch plastic_deformation_rate plastic_spin
              sum_slip_rates slip_rule slip_strength voce_hardening
              integrate_slip_hardening integrate_elastic_strain integrate_orientation'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'slip_rates; elastic_strain slip_hardening orientation'
    residuals = 'slip_rates_residual; elastic_strain_residual slip_hardening_residual orientation_residual'
    structure = 'block dense'
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
  [schur]
    type = SchurComplement
    residual_primary_group = '0'
    unknown_primary_group = '0'
    primary_solver = 'lu'
    schur_solver = 'lu'
  []
[]

[Models]
  [cp_warmup_1]
    type = CrystalPlasticityStrainPredictor
    scale = 0.1
  []
  [cp_warmup_2]
    type = ConstantExtrapolationPredictor
    unknowns_MRP = 'orientation'
    unknowns_Scalar = 'slip_hardening slip_rates'
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
