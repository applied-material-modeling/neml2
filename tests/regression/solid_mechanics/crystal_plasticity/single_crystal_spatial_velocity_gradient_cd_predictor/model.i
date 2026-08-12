# neml2
# single_crystal_spatial_velocity_gradient with the coordinate-descent predictor in the ImplicitUpdate
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
  # L = constant FillR2 with row-major fill
  # [[lxx lxy lxz] [lyx lyy lyz] [lzx lzy lzz]]
  # = [[0.1, 0.01, -0.02], [0.01, -0.05, -0.025], [0.03, -0.01, -0.05]]
  # LinspaceR2(l_single, l_single, 100) expands batched (20,) along a new
  # leading axis of length 100 -> shape (100, 20, 3, 3).
  [L]
    type = Python
    expr = 'R2(torch.tensor([[0.1, 0.01, -0.02], [0.01, -0.05, -0.025], [0.03, -0.01, -0.05]], dtype=torch.float64).reshape(1, 1, 3, 3).expand(100, 20, 3, 3).contiguous())'
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
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model_with_stress'
    prescribed_time = 'times'
    prescribed_R2_names = 'spatial_velocity_gradient'
    prescribed_R2_values = 'L'
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
  [split_to_deformation_rate]
    type = R2ToSR2
    input = 'spatial_velocity_gradient'
    output = 'deformation_rate'
  []
  [split_to_vorticity]
    type = R2ToWR2
    input = 'spatial_velocity_gradient'
    output = 'vorticity'
  []
  [euler_rodrigues]
    type = RotationMatrix
    from = 'orientation'
    to = 'orientation_matrix'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    coefficients = '1e5 0.25'
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
  [implicit_rate]
    type = ComposedModel
    models = 'split_to_deformation_rate split_to_vorticity euler_rodrigues elasticity
              orientation_rate resolved_shear elastic_stretch plastic_deformation_rate
              plastic_spin sum_slip_rates slip_rule slip_strength voce_hardening
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
    max_linesearch_iterations = 5
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  # Blocks whose variable names carry the same value in both graphs are the
  # residual's OWN, listed here rather than duplicated: `plastic_deformation_rate`, `slip_rule`, `sum_slip_rates`.
  # What turns that on is the rotation matrix below taking its plain name: with
  # `orientation_matrix` bound to the OLD orientation, the reused rate leaves pick
  # up the frozen trial frame without a rename. The twins that remain are the ones
  # that genuinely differ -- the trial chain (same leaves, trial values), the
  # lagged strength/hardening leaves (they read `~1`), and the forward-Euler
  # integrators (the residual integrates backward, which is a different model
  # rather than a rename).
  # This scenario prescribes the spatial velocity gradient, not d and w.
  # Redo the same split the implicit model does, so the trial state is real.
  [pred_split_deformation_rate]
    type = R2ToSR2
    input = 'spatial_velocity_gradient'
    output = 'deformation_rate'
  []
  [pred_split_vorticity]
    type = R2ToWR2
    input = 'spatial_velocity_gradient'
    output = 'vorticity'
  []
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
    to = 'orientation_matrix'
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
  # repeatedly at trial driving forces inside the scalar solves. It is the
  # residual's own `slip_rule`, so the exponent and gamma0 are stated once.
  [pred_cd]
    type = CoordinateDescentPredictor
    rate_law = 'slip_rule'
    driving_force_input = 'resolved_shears'
    coupling = 'slip_coupling'
    trial_driving_force = 'resolved_shears_trial'
    rate = 'slip_rates'
    sweeps = 16
  []

  # --- push the predicted rates back onto the unknowns ---
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
  # Orientation is not worth predicting over one step; carried forward.
  [pred_orientation]
    type = ConstantExtrapolationPredictor
    unknowns_MRP = 'orientation'
  []
  [predictor]
    type = ComposedModel
    models = 'pred_split_deformation_rate pred_split_vorticity pred_trial_strain pred_rotation
              pred_trial_stress pred_trial_rss pred_coupling pred_slip_strength pred_cd
              plastic_deformation_rate pred_elastic_rate pred_strain sum_slip_rates
              pred_hardening_rate pred_hardening pred_seed pred_orientation'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [full_stress]
    type = SR2ToR2
    input = 'cauchy_stress'
    output = 'full_cauchy_stress'
  []
  [model_with_stress]
    type = ComposedModel
    models = 'model elasticity full_stress'
    additional_outputs = 'elastic_strain'
  []
[]
