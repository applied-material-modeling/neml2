# neml2
# single_crystal_decoupled_explicit_orientation with the coordinate-descent predictor in the ImplicitUpdate
# predictor slot.
#
# Everything else -- residual, unknowns, load history, solver -- is byte-identical
# to the parent, and the gold is the parent's file COPIED UNCHANGED. That is the
# point: a predictor moves only the initial guess, so passing against the
# parent's reference proves coordinate descent does not perturb the converged
# answer. A failure here means the predictor is changing physics, which no
# predictor is allowed to do.
[Tensors]

  # Mandel isotropic stiffness for SlipSystemElasticInteraction, from the SAME
  # (E, nu) as [Models/elasticity]. Keep the two in step: the coupling matrix is
  # only consistent with the residual if they agree.
  [C_cd_iso]
    type = Python
    expr = 'SSR4((lambda E, nu: (lambda lam, mu: 2.0 * mu * torch.eye(6, dtype=torch.float64) + torch.nn.functional.pad(torch.full((3, 3), lam, dtype=torch.float64), (0, 3, 0, 3)))(E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))))(100000, 0.25))'
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
  # deformation_rate single = FillSR2(0.1, -0.05, -0.05) batched (20,)
  [deformation_rate_single]
    type = Python
    expr = 'SR2.fill(0.1, -0.05, -0.05, 0.0, 0.0, 0.0).dynamic_batch.expand(20)'
  []
  # deformation_rate = LinspaceSR2(d_single, d_single, 100) -> shape (100, 20, 6)
  [deformation_rate]
    type = Python
    expr = 'SR2(deformation_rate_single.data.unsqueeze(0).expand(100, 20, 6).contiguous())'
  []
  # vorticity single = FillWR2(0.1, -0.05, -0.05) batched (20,)
  [vorticity_single]
    type = Python
    expr = 'WR2(torch.tensor([0.1, -0.05, -0.05], dtype=torch.float64).unsqueeze(0).expand(20, 3).contiguous())'
  []
  # vorticity = LinspaceWR2(w_single, w_single, 100) -> shape (100, 20, 3)
  [vorticity]
    type = Python
    expr = 'WR2(vorticity_single.data.unsqueeze(0).expand(100, 20, 3).contiguous())'
  []

  # Crystal geometry inputs.
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
  [initial_orientation]
    type = Python
    expr = 'MRP((lambda r: r / (torch.sqrt((r * r).sum(-1, keepdim=True) + 1.0) + 1.0))(torch.stack([torch.linspace(0.0, 0.75, 20, dtype=torch.float64), torch.linspace(0.0, -0.25, 20, dtype=torch.float64), torch.linspace(-0.1, 0.1, 20, dtype=torch.float64)], dim=-1)))'
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
    ic_MRP_names = 'orientation'
    ic_MRP_values = 'initial_orientation'
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
  ############################################################################
  # Sub-system #1: implicit update of elastic_strain + slip_hardening using
  # LAGGED orientation (orientation~1).
  ############################################################################
  [euler_rodrigues]
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
  [implicit_rate_1]
    type = ComposedModel
    models = 'euler_rodrigues elasticity resolved_shear
              elastic_stretch plastic_deformation_rate
              sum_slip_rates slip_rule slip_strength voce_hardening
              integrate_slip_hardening integrate_elastic_strain'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate_1'
    unknowns = 'elastic_strain slip_hardening'
    residuals = 'elastic_strain_residual slip_hardening_residual'
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
  # --- trial state: elastic strain advanced with no plastic slip ---
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
    elastic_stiffness_tensor = 'C_cd_iso'
    coupling = 'slip_coupling'
  []
  # Slip strengths are LAGGED: the condensed system treats them as frozen over
  # the step, which is the one-round linearization the predictor rests on.
  [pred_slip_strength]
    type = SingleSlipStrengthMap
    slip_hardening = 'slip_hardening~1'
    constant_strength = 50.0
  []

  # --- coordinate descent on phi(gdot) + A gdot = b ---
  # The rate law is a DEPENDENCY, not a graph member: the predictor calls it
  # repeatedly at trial driving forces inside the scalar solves.
  [pred_slip_rule]
    type = PowerLawSlipRule
    n = 8.0
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

  # --- push the predicted rates back onto the unknowns ---
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
  # Cold step -> the coordinate-descent guess; every later step -> the previous
  # converged value. One model, one first-step test, and it is the extrapolator's
  # own: "is there a second history point", which is the actual question. A test
  # on a variable's magnitude only coincides with coldness when that variable
  # starts at zero -- false for a dislocation density or an identity Fp.
  [pred_seed]
    type = ConstantExtrapolationPredictor
    unknowns_SR2 = 'elastic_strain'
    unknowns_Scalar = 'slip_hardening'
    cold = 'elastic_strain:elastic_strain_cd slip_hardening:slip_hardening_cd'
  []
  [predictor]
    type = ComposedModel
    models = 'pred_trial_strain pred_rotation pred_trial_stress pred_trial_rss
              pred_coupling pred_slip_strength pred_cd pred_plastic_rate
              pred_elastic_rate pred_strain
              pred_sum_slip_rates pred_hardening_rate pred_hardening
              pred_seed'
  []
  [subsystem1]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [orientation_rate]
    type = OrientationRate
    elastic_strain = 'elastic_strain'
  []
  [plastic_spin]
    type = PlasticVorticity
  []
  [integrate_orientation]
    type = WR2ExplicitExponentialTimeIntegration
    variable = 'orientation'
  []
  [subsystem2]
    type = ComposedModel
    models = 'euler_rodrigues elasticity resolved_shear
              plastic_deformation_rate plastic_spin
              slip_rule slip_strength orientation_rate
              integrate_orientation'
  []
  [model]
    type = ComposedModel
    models = 'subsystem1 subsystem2'
    additional_outputs = 'elastic_strain slip_hardening'
  []
[]
