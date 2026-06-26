# neml2
# Single-crystal coupled crystal plasticity with multiplicative F = Fe Fp
# decomposition, native port of
# tests/regression/solid_mechanics/crystal_plasticity/single_crystal_coupled_multiplicative/model.i.
# Dynamic batch axis = (20,) (varying end-time / F_end across the batch);
# time axis = (100,). Orientation is a prescribed force, NOT an unknown:
# only ``tauc`` (Scalar) and ``Fp`` (R2) are integrated implicitly.
[Tensors]
  # end_time = LinspaceScalar(1, 10, 20) -> shape (20,)
  [end_time]
    type = Python
    expr = 'Scalar.linspace(1.0, 10.0, 20)'
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
    expr = 'Rot((lambda r: r / (torch.sqrt((r * r).sum(-1, keepdim=True) + 1.0) + 1.0))(torch.stack([torch.linspace(0.0, 0.75, 20, dtype=torch.float64), torch.linspace(0.0, -0.25, 20, dtype=torch.float64), torch.linspace(-0.1, 0.1, 20, dtype=torch.float64)], dim=-1)))'
  []
  # r = LinspaceRot(initial_orientation, initial_orientation, 100) -> shape (100, 20, 3)
  [r]
    type = Python
    expr = 'Rot(initial_orientation.data.unsqueeze(0).expand(100, 20, 3).contiguous())'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model_with_pk2_stress'
    prescribed_time = 'times'
    prescribed_R2_names = 'F'
    prescribed_R2_values = 'F'
    prescribed_Rot_names = 'r'
    prescribed_Rot_values = 'r'
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
    models = 'euler_rodrigues slip_strength voce_hardening
              elasticity resolved_shear slip_rule sum_slip_rates
              plastic_velgrad plastic_defgrad_rate
              integrate_slip_hardening integrate_plastic_defgrad'
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
  [predictor]
    type = LinearExtrapolationPredictor
    unknowns_Scalar = 'tauc'
    unknowns_R2 = 'Fp'
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
