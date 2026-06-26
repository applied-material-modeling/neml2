# neml2
# Single-crystal coupled crystal plasticity with multiplicative F = Fe Fp
# decomposition. Mirrors tests/regression/.../single_crystal_coupled_multiplicative
# but parameterised over ``${nbatch}`` and ``${device}`` so it slots into
# the benchmark sweep.
#
# This benchmark uses plain ``Newton`` (no line search) while the rest of
# the CP suite (scpdecoup, scpcoup, scpdecoupexp, tcpsingle, tcprandom)
# uses ``NewtonWithLineSearch``. Comparing scpcoupmult against the others
# isolates the cost of line search inside the implicit segment.
[Settings]
  example_batch_shape = '(${nbatch},)'
[]

[Tensors]
  # HIT-substituted shim so the verbatim triple-quoted Python blocks below
  # can reference nbatch as a bare identifier.
  [nbatch]
    type = Python
    expr = '${nbatch}'
  []
  # end_time = LinspaceScalar(1, 10, nbatch) -> shape (nbatch,)
  [end_time]
    type = Python
    expr = '''
      Scalar(torch.linspace(1.0, 10.0, nbatch, dtype=torch.float64))
    '''
  []
  # times = LinspaceScalar(0, end_time, 100) -> shape (100, nbatch)
  [times]
    type = Python
    expr = '''
      Scalar(
          end_time.data.unsqueeze(0)
          * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1)
      )
    '''
  []

  # Deformation gradient F: time-varying, batch-varying.
  # F_start = identity (3,3); F_end varies across the batch from a small
  # perturbation to a larger one. F[k, b] = F_start + (k/99) * (F_end[b] - F_start)
  # -> shape (100, nbatch, 3, 3).
  [F]
    type = Python
    expr = '''
F_start = torch.eye(3, dtype=torch.float64)
F_end_min = torch.tensor(
    [[1.005, 0.001, 0.005], [0.001, 0.991, -0.03], [-0.005, 0.002, 1.008]],
    dtype=torch.float64,
)
F_end_max = torch.tensor(
    [[1.05, 0.01, 0.05], [0.01, 0.91, -0.3], [-0.05, 0.02, 1.08]],
    dtype=torch.float64,
)
F_end = (
    F_end_min.unsqueeze(0)
    + torch.linspace(0.0, 1.0, nbatch, dtype=torch.float64).reshape(nbatch, 1, 1)
    * (F_end_max - F_end_min).unsqueeze(0)
)
F_full = (
    F_start.reshape(1, 1, 3, 3)
    + torch.linspace(0.0, 1.0, 100, dtype=torch.float64).reshape(100, 1, 1, 1)
    * (F_end.reshape(1, nbatch, 3, 3) - F_start.reshape(1, 1, 3, 3))
)
result = R2(F_full.contiguous())
'''
  []

  # Initial plastic deformation gradient = identity, shape (3, 3) (no batch).
  [Fp0]
    type = Python
    expr = 'R2.identity()'
  []

  # Crystal geometry inputs.
  [a]
    type = Python
    expr = '''
      Scalar(torch.tensor([1.0], dtype=torch.float64))
    '''
  []
  [sdirs]
    type = Python
    expr = '''
      MillerIndex(torch.tensor([1, 1, 0], dtype=torch.int64))
    '''
  []
  [splanes]
    type = Python
    expr = '''
      MillerIndex(torch.tensor([1, 1, 1], dtype=torch.int64))
    '''
  []

  # Initial orientation = FillRot(R1, R2, R3, method='standard'):
  # convert standard Rodrigues r_std to modified-Rodrigues parameters via
  # r = r_std / (sqrt(|r_std|^2 + 1) + 1). Shape (nbatch, 3).
  # R1 = linspace(0, 0.75, nbatch); R2 = linspace(0, -0.25, nbatch); R3 = linspace(-0.1, 0.1, nbatch).
  [initial_orientation]
    type = Python
    expr = '''
      Rot(
          (lambda r: r / (torch.sqrt((r * r).sum(-1, keepdim=True) + 1.0) + 1.0))(
              torch.stack(
                  [
                      torch.linspace(0.0, 0.75, nbatch, dtype=torch.float64),
                      torch.linspace(0.0, -0.25, nbatch, dtype=torch.float64),
                      torch.linspace(-0.1, 0.1, nbatch, dtype=torch.float64),
                  ],
                  dim=-1,
              )
          )
      )
    '''
  []
  # r = LinspaceRot(initial_orientation, initial_orientation, 100) -> shape (100, nbatch, 3)
  [r]
    type = Python
    expr = '''
      Rot(initial_orientation.data.unsqueeze(0).expand(100, nbatch, 3).contiguous())
    '''
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
  # Plain Newton (no line search) -- the whole point of this scenario,
  # vs. the rest of the CP suite which all use NewtonWithLineSearch.
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
