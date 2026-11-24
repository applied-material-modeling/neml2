# neml2
# Exact single-crystal kinematics: integrate the plastic deformation gradient Fp
# directly from the plastic spatial velocity gradient Lp using the full F = Fe Fp
# multiplicative decomposition (no small-strain approximation), with a
# St. Venant-Kirchhoff elastic law on the Green-Lagrange strain of Fe = F Fp^-1.
# The orientation enters as a fixed reference force `r`; only `tauc` (slip
# hardening) and `Fp` are integrated implicitly. Driven by pyzag in
# formulations.ipynb.
[Tensors]
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
  # Orientation remains constant in the reference configuration: convert the
  # modified Rodrigues parameters `r` to the rotation matrix R.
  [euler_rodrigues]
    type = RotationMatrix
    from = 'r'
    to = 'R'
  []
  # Hardening (very simple)
  [slip_strength]
    type = SingleSlipStrengthMap
    constant_strength = 180.0
    slip_hardening = 'tauc'
    slip_strengths = 'tauc_i'
  []
  [voce_hardening]
    type = VoceSingleSlipHardeningRule
    initial_slope = 2000.0
    saturated_hardening = 500.0
    slip_hardening = 'tauc'
    sum_slip_rates = 'gamma_rate'
    slip_hardening_rate = 'tauc_rate'
  []
  # Elasticity: St. Venant-Kirchhoff with the Green-Lagrange strain of Fe.
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
  [elastic_tensor]
    type = CubicElasticityTensor
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO SHEAR_MODULUS'
    coefficients = '209016 0.307 60355.0'
  []
  [svk]
    type = GeneralElasticity
    elastic_stiffness_tensor = 'elastic_tensor'
    strain = 'E'
    stress = 'S'
    orientation = 'r'
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
    n = 6
    gamma0 = 0.0001
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
