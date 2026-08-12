# neml2
# single_crystal_coupled with the coordinate-descent predictor in the
# ImplicitUpdate predictor slot instead of CrystalPlasticityStrainPredictor.
#
# Everything else -- residual, unknowns, load history, solver -- is byte-identical
# to the parent, and the gold is the parent's file COPIED UNCHANGED. That is the
# whole point of the scenario: a predictor moves only the initial guess, so if
# this passes against the parent's reference then coordinate descent provably
# does not perturb the converged answer. If it ever fails, the predictor is
# changing physics, which no predictor is allowed to do.
#
# The Newton counts it saves are measured in studies/nlprecond; this file only
# pins that the answer is unmoved.
[Tensors]
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
  # deformation_rate single = FillSR2(dxx=0.1, dyy=-0.05, dzz=-0.05) batched (20,)
  [deformation_rate_single]
    type = Python
    expr = 'SR2.fill(0.1, -0.05, -0.05, 0.0, 0.0, 0.0).dynamic_batch.expand(20)'
  []
  # deformation_rate = LinspaceSR2(d_single, d_single, 100) -> shape (100, 20, 6)
  [deformation_rate]
    type = Python
    expr = 'SR2(deformation_rate_single.data.unsqueeze(0).expand(100, 20, 6).contiguous())'
  []
  # vorticity single = FillWR2(w1=0.1, w2=-0.05, w3=-0.05) batched (20,)
  [vorticity_single]
    type = Python
    expr = 'WR2(torch.tensor([0.1, -0.05, -0.05], dtype=torch.float64).unsqueeze(0).expand(20, 3).contiguous())'
  []
  # vorticity = LinspaceWR2(w_single, w_single, 100) -> shape (100, 20, 3)
  [vorticity]
    type = Python
    expr = 'WR2(vorticity_single.data.unsqueeze(0).expand(100, 20, 3).contiguous())'
  []

  # Mandel isotropic stiffness for SlipSystemElasticInteraction, from the SAME
  # (E, nu) as [Models/elasticity]. Keep the two in step: the coupling matrix is
  # only consistent with the residual if they agree.
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
  # r = r_std / (sqrt(|r_std|^2 + 1) + 1). Shape (20, 3).
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
    elastic_stiffness_tensor = 'C_iso'
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
  [pred_strain_gate]
    type = SR2MagnitudeGate
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
    type = ScalarMagnitudeGate
    prediction = 'slip_hardening_cd'
    reference = 'slip_hardening~1'
    gated = 'slip_hardening'
    threshold = 1e-3
  []
  # Orientation measured as not worth predicting over one step; carried forward.
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
