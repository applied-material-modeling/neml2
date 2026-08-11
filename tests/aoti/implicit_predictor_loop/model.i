# neml2
# Crystal plasticity with a CoordinateDescentPredictor, for the AOTI
# predictor-loop tests.
#
# The predictor is a bounded Gauss-Seidel iteration. Exporting it as-is would
# unroll every sweep into the graph, so the exporter emits ONE sweep plus a
# feedback pair and the C++ runtime drives the loop. This fixture is the
# smallest wiring that exercises that path end to end: trial state -> coupling
# -> coordinate descent -> back-substitution -> cold-state gate.
[Tensors]
  # end_time = LinspaceScalar(1, 10, nbatch) -> shape (nbatch,)
  [end_time]
    type = Python
    expr = 'linspace(Scalar(1.0).dynamic_batch, Scalar(10.0).dynamic_batch, 2)'
  []
  # times = LinspaceScalar(0, end_time, npoint) -> shape (npoint, nbatch)
  [times]
    type = Python
    expr = 'Scalar(end_time.data.unsqueeze(0) * torch.linspace(0.0, 0.010101010101010102, 2, dtype=torch.float64).unsqueeze(-1))'
  []
  # deformation_rate single = FillSR2(dxx=0.1, dyy=-0.05, dzz=-0.05) batched (nbatch,)
  [deformation_rate_single]
    type = Python
    expr = 'SR2.fill(0.1, -0.05, -0.05, 0.0, 0.0, 0.0).dynamic_batch.expand(2)'
  []
  # deformation_rate = LinspaceSR2(d_single, d_single, npoint) -> shape (npoint, nbatch, 6)
  [deformation_rate]
    type = Python
    expr = 'SR2(deformation_rate_single.data.unsqueeze(0).expand(2, 2, 6).contiguous())'
  []
  # vorticity single = FillWR2(w1=0.1, w2=-0.05, w3=-0.05) batched (nbatch,)
  [vorticity_single]
    type = Python
    expr = 'WR2(torch.tensor([0.1, -0.05, -0.05], dtype=torch.float64).unsqueeze(0).expand(2, 3).contiguous())'
  []
  # vorticity = LinspaceWR2(w_single, w_single, npoint) -> shape (npoint, nbatch, 3)
  [vorticity]
    type = Python
    expr = 'WR2(vorticity_single.data.unsqueeze(0).expand(2, 2, 3).contiguous())'
  []

  # Mandel isotropic stiffness for SlipSystemElasticInteraction, built from the
  # SAME (E, nu) as [Models/elasticity] below. Keep the two in step: the
  # coupling matrix is only consistent with the residual if they agree.
  [C_iso]
    type = Python
    expr = 'SSR4((lambda E, nu: (lambda lam, mu: 2.0 * mu * torch.eye(6, dtype=torch.float64) + torch.nn.functional.pad(torch.full((3, 3), lam, dtype=torch.float64), (0, 3, 0, 3)))(E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))))(1e5, 0.25))'
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
    expr = 'MRP((lambda r: r / (torch.sqrt((r * r).sum(-1, keepdim=True) + 1.0) + 1.0))(torch.stack([torch.linspace(0.0, 0.75, 2, dtype=torch.float64), torch.linspace(0.0, -0.25, 2, dtype=torch.float64), torch.linspace(-0.1, 0.1, 2, dtype=torch.float64)], dim=-1)))'
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
    type = PowerLawSlipRule
    n = '8.0'
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
    unknowns = 'elastic_strain slip_hardening orientation'
    residuals = 'elastic_strain_residual slip_hardening_residual orientation_residual'
  []
[]

[Solvers]
  [newton]
    type = NewtonWithLineSearch
    max_linesearch_iterations = '5'
    max_its = '50'
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  # ------------------------------------------------------------------
  # Coordinate-descent predictor.
  #
  # Stage 1 -- the condensed system's data:
  #   b = trial resolved shear, i.e. the driving force with every slip rate at
  #       zero: elastic strain advanced by the deformation rate alone.
  #   A = dt * M^T C M, the elastic interaction between slip systems.
  # Stage 2 -- coordinate descent solves phi(gdot) + A gdot = b for the rates.
  # Stage 3 -- push the rates back onto the unknowns the solve actually carries.
  # ------------------------------------------------------------------

  # --- stage 1: trial state ---------------------------------------------
  [pred_trial_strain]
    type = SR2ForwardEulerTimeIntegration
    variable = 'elastic_strain_trial'
    old_variable = 'elastic_strain~1'
    rate = 'deformation_rate'
  []
  [pred_rotation]
    type = RotationMatrix
    from = 'orientation~1'
    to = 'orientation_matrix_old'
  []
  [pred_trial_stress]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.25'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain_trial'
    stress = 'cauchy_stress_trial'
  []
  [pred_trial_rss]
    type = ResolvedShear
    stress = 'cauchy_stress_trial'
    orientation_matrix = 'orientation_matrix_old'
    resolved_shears = 'resolved_shears_trial'
  []
  [pred_coupling]
    type = SlipSystemElasticInteraction
    orientation = 'orientation~1'
    elastic_stiffness_tensor = 'C_iso'
    coupling = 'slip_coupling'
  []
  # Slip strengths are LAGGED -- evaluated at the old hardening. The condensed
  # system treats them as frozen over the step; that is the one-round
  # linearization the predictor is built on.
  [pred_slip_strength]
    type = SingleSlipStrengthMap
    slip_hardening = 'slip_hardening~1'
    constant_strength = 50.0
  []

  # --- stage 2: coordinate descent --------------------------------------
  # The rate law is a DEPENDENCY, not a graph member: the predictor calls it
  # repeatedly at trial driving forces during the inner scalar solves.
  [pred_slip_rule]
    type = PowerLawSlipRule
    n = '8.0'
    gamma0 = 2.0e-1
  []
  [pred_cd]
    type = CoordinateDescentPredictor
    rate_law = 'pred_slip_rule'
    driving_force_input = 'resolved_shears'
    coupling = 'slip_coupling'
    trial_driving_force = 'resolved_shears_trial'
    rate = 'slip_rates'
    sweeps = 16
  []

  # --- stage 3: back-substitution onto the unknowns ---------------------
  [pred_plastic_rate]
    type = PlasticDeformationRate
    orientation_matrix = 'orientation_matrix_old'
  []
  [pred_elastic_rate]
    type = ElasticStrainRate
    elastic_strain = 'elastic_strain~1'
  []
  [pred_strain]
    type = SR2ForwardEulerTimeIntegration
    variable = 'elastic_strain_cd'
    old_variable = 'elastic_strain~1'
    rate = 'elastic_strain_rate'
  []
  # Cold-state gate: the prediction is worth having only on the step with no
  # previous solution to start from. At warm steps the previous converged value
  # is already the better guess, so it passes through untouched.
  [pred_strain_gate]
    type = SR2ColdStateGate
    prediction = 'elastic_strain_cd'
    reference = 'elastic_strain~1'
    gated = 'elastic_strain'
    threshold = 1e-3
  []
  [pred_sum_slip_rates]
    type = SumSlipRates
  []
  [pred_hardening_rate]
    type = VoceSingleSlipHardeningRule
    slip_hardening = 'slip_hardening~1'
    initial_slope = 500.0
    saturated_hardening = 50.0
  []
  [pred_hardening]
    type = ScalarForwardEulerTimeIntegration
    variable = 'slip_hardening_cd'
    old_variable = 'slip_hardening~1'
    rate = 'slip_hardening_rate'
  []
  [pred_hardening_gate]
    type = ScalarColdStateGate
    prediction = 'slip_hardening_cd'
    reference = 'slip_hardening~1'
    gated = 'slip_hardening'
    threshold = 1e-3
  []
  # Orientation is not worth predicting over one step -- measured as worth
  # nothing -- so it is simply carried forward.
  [pred_orientation]
    type = ConstantExtrapolationPredictor
    unknowns_MRP = 'orientation'
  []

  [predictor]
    type = ComposedModel
    models = 'pred_trial_strain pred_rotation pred_trial_stress pred_trial_rss
              pred_coupling pred_slip_strength pred_cd pred_plastic_rate
              pred_elastic_rate pred_strain pred_strain_gate
              pred_sum_slip_rates pred_hardening_rate pred_hardening
              pred_hardening_gate pred_orientation'
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
