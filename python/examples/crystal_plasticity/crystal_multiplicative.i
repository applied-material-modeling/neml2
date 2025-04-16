[Tensors]
  [a]
    type = Scalar
    values = '1.0'
  []
  [sdirs]
    type = FillMillerIndex
    values = '1 1 0'
  []
  [splanes]
    type = FillMillerIndex
    values = '1 1 1'
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
  # Orientation remains constant as we work with the reference configuration
  [euler_rodrigues]
    type = RotationMatrix
    from = 'forces/r'
    to = 'forces/R'
  []
  # Hardening (this is just a very simple hardening model)
  [slip_strength]
    type = SingleSlipStrengthMap
    constant_strength = 180.0
    slip_hardening = 'state/tauc'
    slip_strengths = 'state/tauc_i'
  []
  [voce_hardening]
    type = VoceSingleSlipHardeningRule
    initial_slope = 2000.0
    saturated_hardening = 500.0
    slip_hardening_rate = 'state/tauc_rate'
    slip_hardening = 'state/tauc'
    sum_slip_rates = 'state/gamma_rate'
  []
  # Elasticity: St. Venant-Kirchhoff with Green-Lagrange strain
  [mult_decomp]
    type = R2Multiplication
    A = 'forces/F'
    B = 'state/Fp'
    to = 'state/Fe'
    invert_B = true
  []
  [gl_strain]
    type = GreenLagrangeStrain
    deformation_gradient = 'state/Fe'
    strain = 'state/E'
  []
  [elastic_tensor]
    type = CubicElasticityTensor
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO SHEAR_MODULUS'
    coefficients = '209016 0.307 60355.0'
  []
  [svk]
    type = GeneralElasticity
    elastic_stiffness_tensor = 'elastic_tensor'
    strain = 'state/E'
    stress = 'state/S'
    orientation = 'forces/r'
  []
  [elasticity]
    type = ComposedModel
    models = 'mult_decomp gl_strain svk'
  []
  # CP flow rule
  [resolved_shear]
    type = ResolvedShear
    resolved_shears = 'state/tau_i'
    stress = 'state/S'
    orientation = 'forces/R'
  []
  [slip_rule]
    type = PowerLawSlipRule
    n = 6
    gamma0 = 0.0001
    slip_rates = 'state/gamma_rate_i'
    resolved_shears = 'state/tau_i'
    slip_strengths = 'state/tauc_i'
  []
  [sum_slip_rates]
    type = SumSlipRates
    slip_rates = 'state/gamma_rate_i'
    sum_slip_rates = 'state/gamma_rate'
  []
  [plastic_velgrad_sym]
    type = PlasticDeformationRate
    plastic_deformation_rate = 'state/Lp_sym'
    slip_rates = 'state/gamma_rate_i'
    orientation = 'forces/R'
  []
  [plastic_velgrad]
    type = SR2toR2
    input = 'state/Lp_sym'
    output = 'state/Lp'
  []
  [plastic_defgrad_rate]
    type = R2Multiplication
    A = 'state/Lp'
    B = 'state/Fp'
    to = 'state/Fp_rate'
  []
  # Definition of residuals
  [integrate_slip_hardening]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/tauc'
  []
  [integrate_plastic_defgrad]
    type = R2BackwardEulerTimeIntegration
    variable = 'state/Fp'
  []
  # The implicit model that we solve for
  [implicit_rate]
    type = ComposedModel
    models = "euler_rodrigues slip_strength voce_hardening
              elastic_tensor mult_decomp gl_strain svk
              resolved_shear slip_rule sum_slip_rates
              plastic_velgrad_sym plastic_velgrad plastic_defgrad_rate
              integrate_slip_hardening integrate_plastic_defgrad"
  []
[]
