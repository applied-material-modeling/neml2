# neml2
# single_crystal_coupled_multiplicative with the coordinate-descent predictor in the ImplicitUpdate
# predictor slot, replacing the LinearExtrapolationPredictor.
#
# The gold is the parent's file COPIED UNCHANGED: a predictor moves only the
# initial guess, so passing against the parent's reference proves coordinate
# descent does not perturb the converged answer.
#
# THIS SCENARIO IS NOT LIKE THE SMALL-STRAIN ONES, in three ways.
#
# 1. The trial state is EXACT rather than a forward-Euler guess. At zero slip Fp
#    stays at Fp~1, so Fe_trial = F * Fp~1^-1 uses the CURRENT deformation
#    gradient -- no linearization in the trial at all.
#
# 2. The coupling matrix is unchanged. With Fe = F Fp^-1 and Fp_dot = Lp Fp,
#    dFe = -Fe (dt Lp), so dE = -dt sym(Fe^T Fe sum_j gdot_j M_j) and, for
#    Fe^T Fe ~ I (elastic strains here are ~1e-3), dtau_i = -dt M_i : C : M_j
#    gdot_j. That is the same A = dt M^T C M the small-strain scenarios use, with
#    a neglected term of order the elastic strain -- SMALLER than the spin term
#    already documented on SlipSystemElasticInteraction.
#
# 3. The cold/warm switch is the EXTRAPOLATOR'S, not a separate gate. Fp starts
#    at the IDENTITY, so a gate on its own magnitude is shut forever; and this
#    scenario's warm steps need linear extrapolation, not a frozen guess, so
#    falling back to `u~1` stalls it (its sibling's header says as much). The
#    LinearExtrapolationPredictor already tests for "no history to extrapolate
#    from" via |t~1 - t~2|, so it simply takes the coordinate-descent values as
#    its cold branch through `cold_value_suffix`.
#
# Orientation is a prescribed force here (`r`), not an unknown, so the predictor
# reads it directly rather than through a `~1` lag.
[Tensors]

  # Mandel isotropic stiffness for SlipSystemElasticInteraction, from the SAME
  # (E, nu) as [Models/svk]. Keep the two in step: the coupling matrix is only
  # consistent with the residual if they agree.
  [C_cd_iso]
    type = Python
    expr = 'SSR4((lambda E, nu: (lambda lam, mu: 2.0 * mu * torch.eye(6, dtype=torch.float64) + torch.nn.functional.pad(torch.full((3, 3), lam, dtype=torch.float64), (0, 3, 0, 3)))(E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))))(1e5, 0.25))'
  []
  # end_time = LinspaceScalar(1, 10, 20) -> shape (20,)
  [end_time]
    type = Python
    expr = 'linspace(Scalar(1.0).dynamic_batch, Scalar(10.0).dynamic_batch, 20)'
  []
  # times = LinspaceScalar(0, end_time, 100) -> shape (100, 20)
  [times]
    type = Python
    expr = 'Scalar(end_time.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1))'
  []

  # F_start = identity (3,3)
  # F_end_min = '1.005 0.001 0.005  0.001 0.991 -0.03  -0.005 0.002 1.008'
  # F_end_max = '1.05  0.01  0.05   0.01  0.91  -0.3   -0.05  0.02  1.08'
  # F_end = LinspaceR2(F_end_min, F_end_max, 20) -> shape (20, 3, 3)
  # F = LinspaceR2(F_start, F_end, 100) -> shape (100, 20, 3, 3)
  #   F[k, b] = F_start + (k / 99) * (F_end[b] - F_start)
  [F]
    type = Python
    expr = 'F_start = torch.eye(3, dtype=torch.float64)
F_end_min = torch.tensor([[1.005, 0.001, 0.005], [0.001, 0.991, -0.03], [-0.005, 0.002, 1.008]], dtype=torch.float64)
F_end_max = torch.tensor([[1.05, 0.01, 0.05], [0.01, 0.91, -0.3], [-0.05, 0.02, 1.08]], dtype=torch.float64)
F_end = F_end_min.unsqueeze(0) + torch.linspace(0.0, 1.0, 20, dtype=torch.float64).reshape(20, 1, 1) * (F_end_max - F_end_min).unsqueeze(0)
F_full = F_start.reshape(1, 1, 3, 3) + torch.linspace(0.0, 1.0, 100, dtype=torch.float64).reshape(100, 1, 1, 1) * (F_end.reshape(1, 20, 3, 3) - F_start.reshape(1, 1, 3, 3))
result = R2(F_full.contiguous())'
  []

  # Initial plastic deformation gradient = identity, shape (3, 3) (no batch)
  [Fp0]
    type = Python
    expr = 'R2.identity()'
  []

  # Crystal geometry inputs
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
  # r = r_std / (sqrt(|r_std|^2 + 1) + 1). Shape (20, 3).
  # R1 = linspace(0, 0.75, 20); R2 = linspace(0, -0.25, 20); R3 = linspace(-0.1, 0.1, 20).
  [initial_orientation]
    type = Python
    expr = 'MRP((lambda r: r / (torch.sqrt((r * r).sum(-1, keepdim=True) + 1.0) + 1.0))(torch.stack([torch.linspace(0.0, 0.75, 20, dtype=torch.float64), torch.linspace(0.0, -0.25, 20, dtype=torch.float64), torch.linspace(-0.1, 0.1, 20, dtype=torch.float64)], dim=-1)))'
  []
  # r = LinspaceRot(initial_orientation, initial_orientation, 100) -> shape (100, 20, 3)
  [r]
    type = Python
    expr = 'MRP(initial_orientation.data.unsqueeze(0).expand(100, 20, 3).contiguous())'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model_with_pk2_stress'
    prescribed_time = 'times'
    prescribed_R2_names = 'F'
    prescribed_R2_values = 'F'
    prescribed_MRP_names = 'r'
    prescribed_MRP_values = 'r'
    ic_R2_names = 'Fp'
    ic_R2_values = 'Fp0'
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
  # Orientation remains constant; convert modified Rodrigues to the rotation matrix R.
  [euler_rodrigues]
    type = RotationMatrix
    from = 'r'
    to = 'R'
  []
  # Hardening (very simple)
  [slip_strength]
    type = SingleSlipStrengthMap
    constant_strength = 50.0
    slip_hardening = 'tauc'
    slip_strengths = 'tauc_i'
  []
  [voce_hardening]
    type = VoceSingleSlipHardeningRule
    initial_slope = 500.0
    saturated_hardening = 50.0
    slip_hardening = 'tauc'
    sum_slip_rates = 'gamma_rate'
    # Native does not auto-derive output names from input renames; provide
    # ``tauc_rate`` explicitly so it matches the integrator's expectation.
    slip_hardening_rate = 'tauc_rate'
  []
  # Elasticity: St. Venant-Kirchhoff with Green-Lagrange strain
  [mult_decomp]
    type = R2Multiplication
    A = 'F'
    B = 'Fp'
    to = 'Fe'
    invert_B = true
  []
  [gl_strain]
    type = GreenLagrangeStrain
    deformation_gradient = 'Fe'
    strain = 'E'
  []
  [svk]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.25'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'E'
    stress = 'S'
  []
  [elasticity]
    type = ComposedModel
    models = 'mult_decomp gl_strain svk'
  []
  # CP flow rule
  [resolved_shear]
    type = ResolvedShear
    resolved_shears = 'tau_i'
    stress = 'S'
    orientation_matrix = 'R'
  []
  [slip_rule]
    type = PowerLawSlipRule
    n = 8.0
    gamma0 = 2.0e-1
    slip_rates = 'gamma_rate_i'
    resolved_shears = 'tau_i'
    slip_strengths = 'tauc_i'
  []
  [sum_slip_rates]
    type = SumSlipRates
    slip_rates = 'gamma_rate_i'
    sum_slip_rates = 'gamma_rate'
  []
  [plastic_velgrad]
    type = PlasticSpatialVelocityGradient
    plastic_spatial_velocity_gradient = 'Lp'
    slip_rates = 'gamma_rate_i'
    orientation_matrix = 'R'
  []
  [plastic_defgrad_rate]
    type = R2Multiplication
    A = 'Lp'
    B = 'Fp'
    to = 'Fp_rate'
  []
  # Residuals
  [integrate_slip_hardening]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'tauc'
  []
  [integrate_plastic_defgrad]
    type = R2BackwardEulerTimeIntegration
    variable = 'Fp'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'euler_rodrigues slip_strength voce_hardening elasticity resolved_shear slip_rule
              sum_slip_rates plastic_velgrad plastic_defgrad_rate integrate_slip_hardening
              integrate_plastic_defgrad'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'tauc Fp'
    residuals = 'tauc_residual Fp_residual'
  []
[]

[Solvers]
  [newton]
    type = Newton
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  # Blocks whose variable names carry the same value in both graphs are the
  # residual's OWN, listed here rather than duplicated: `euler_rodrigues`,
  # `plastic_velgrad`, `slip_rule`, `sum_slip_rates`.
  # The orientation is a given here rather than an unknown, so the rotation matrix
  # is the same value in both graphs and is reused outright. The twins that remain
  # are the ones
  # that genuinely differ -- the trial chain (same leaves, trial values), the
  # lagged strength/hardening leaves (they read `~1`), and the forward-Euler
  # integrators (the residual integrates backward, which is a different model
  # rather than a rename).
  # --- trial state: Fp frozen at its old value, so Fe_trial = F * Fp~1^-1 ---
  [pred_trial_Fe]
    type = R2Multiplication
    A = 'F'
    B = 'Fp~1'
    to = 'Fe_trial'
    invert_B = true
  []
  [pred_trial_E]
    type = GreenLagrangeStrain
    deformation_gradient = 'Fe_trial'
    strain = 'E_trial'
  []
  [pred_trial_S]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.25'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'E_trial'
    stress = 'S_trial'
  []
  [pred_trial_rss]
    type = ResolvedShear
    stress = 'S_trial'
    orientation_matrix = 'R'
    resolved_shears = 'tau_i_trial'
  []
  [pred_coupling]
    type = SlipSystemElasticInteraction
    orientation = 'r'
    elastic_stiffness_tensor = 'C_cd_iso'
    coupling = 'slip_coupling'
  []
  # Slip strengths are LAGGED at the old hardening: the condensed system treats
  # them as frozen over the step.
  [pred_slip_strength]
    type = SingleSlipStrengthMap
    constant_strength = 50.0
    slip_hardening = 'tauc~1'
    slip_strengths = 'tauc_i'
  []

  # --- coordinate descent on phi(gdot) + A gdot = b ---
  [pred_cd]
    type = CoordinateDescentPredictor
    rate_law = 'slip_rule'
    driving_force_input = 'tau_i'
    coupling = 'slip_coupling'
    trial_driving_force = 'tau_i_trial'
    rate = 'gamma_rate_i'
    sweeps = 16
  []

  # --- push the predicted rates back onto Fp and tauc ---
  [pred_Fp_rate]
    type = R2Multiplication
    A = 'Lp'
    B = 'Fp~1'
    to = 'Fp_pred_rate'
  []
  [pred_Fp]
    type = R2ForwardEulerTimeIntegration
    variable = 'Fp_cd'
    old_variable = 'Fp~1'
    rate = 'Fp_pred_rate'
  []
  [pred_tauc_rate]
    type = VoceSingleSlipHardeningRule
    initial_slope = 500.0
    saturated_hardening = 50.0
    slip_hardening = 'tauc~1'
    sum_slip_rates = 'gamma_rate'
    slip_hardening_rate = 'tauc_pred_rate'
  []
  [pred_tauc]
    type = ScalarForwardEulerTimeIntegration
    variable = 'tauc_cd'
    old_variable = 'tauc~1'
    rate = 'tauc_pred_rate'
  []
  # Cold step -> the coordinate-descent guess; every later step -> linear
  # extrapolation, which is what the parent uses and what this scenario needs
  # (a mixed-stiffness batch stalls on a frozen guess). One model, one
  # first-step test -- the extrapolator already had it.
  [pred_extrapolate]
    type = LinearExtrapolationPredictor
    unknowns_Scalar = 'tauc'
    unknowns_R2 = 'Fp'
    cold = 'tauc:tauc_cd Fp:Fp_cd'
  []
  [predictor]
    type = ComposedModel
    models = 'pred_trial_Fe pred_trial_E pred_trial_S euler_rodrigues pred_trial_rss pred_coupling
              pred_slip_strength pred_cd plastic_velgrad pred_Fp_rate pred_Fp sum_slip_rates
              pred_tauc_rate pred_tauc pred_extrapolate'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model_with_pk2_stress]
    type = ComposedModel
    models = 'model elasticity'
    additional_outputs = 'Fp'
  []
[]
