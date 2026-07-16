# neml2
# Single-crystal crystal plasticity with multiplicative F = Fe Fp decomposition,
# exercising the CrystalPlasticityDeformationGradientPredictor (Fp warm-up) in the
# ImplicitUpdate predictor slot. Same building blocks as the sibling
# ``single_crystal_coupled_multiplicative`` scenario, but a single element under a
# few MODERATE steps -- the regime the Fp predictor targets: the frozen-Fp elastic
# trial over-stresses each step, so the predictor's relaxation branch engages and
# the local Newton converges from the seeded low-stress state. (The 20-element,
# 100-step sibling needs linear extrapolation -- a mixed-stiffness batch stalls a
# frozen/relaxed guess -- so this scenario is deliberately homogeneous.)
# The predictor covers only ``Fp``; ``tauc`` falls back to its state value
# (ImplicitUpdate._initial_unknowns). Orientation is a prescribed force, NOT an
# unknown: only ``tauc`` (Scalar) and ``Fp`` (R2) are integrated implicitly.
[Tensors]
  # Single element, few MODERATE steps: each increment is large enough that the
  # frozen-Fp elastic trial over-stresses (trial elastic strain >> yield), which
  # is exactly the regime the Fp warm-up predictor targets -- so its relaxation
  # branch engages and the local Newton converges from the seeded low-stress
  # state. A homogeneous single element avoids the mixed-stiffness batched-Newton
  # stall (the 100-step, 20-element sibling needs linear extrapolation instead).
  #
  # times = linspace(0, 1, 11) -> 10 steps, shape (11,)
  [times]
    type = Python
    expr = 'Scalar(torch.linspace(0.0, 1.0, 11, dtype=torch.float64))'
  []
  # F = linspace(I, F_end, 11) -> shape (11, 3, 3); ~4% stretch + 6% shear.
  [F]
    type = Python
    expr = 'I3 = torch.eye(3, dtype=torch.float64)
F_end = torch.tensor([[1.04, 0.0, 0.06], [0.0, 0.98, 0.0], [0.0, 0.0, 0.99]], dtype=torch.float64)
s = torch.linspace(0.0, 1.0, 11, dtype=torch.float64).reshape(11, 1, 1)
result = R2((I3.reshape(1, 3, 3) + s * (F_end - I3).reshape(1, 3, 3)).contiguous())'
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

  # Fixed single orientation (modified Rodrigues), constant over time: shape (11, 3).
  [r]
    type = Python
    expr = 'MRP(torch.tensor([0.15, -0.08, 0.03], dtype=torch.float64).reshape(1, 3).expand(11, 3).contiguous())'
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
    type = CrystalPlasticityDeformationGradientPredictor
    deformation_gradient = 'F'
    plastic_deformation_gradient = 'Fp'
    scale = 0.1
    threshold = 1e-3
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
