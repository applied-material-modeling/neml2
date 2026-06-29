# neml2
## Applying KKT conditions with the help of Fisher-Burmeister complementary condition

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_SR2_names = 'E'
    prescribed_SR2_values = 'strains'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Tensors]
  [times]
    type = Python
    expr = 'linspace(Scalar(0.0).dynamic_batch, Scalar(3.0).dynamic_batch, 1000)'
  []
  [strains]
    # LinspaceSR2 0 -> (0.016, -0.008, -0.008, 0, 0, 0) across 1000 steps -> (1000, 6).
    type = Python
    expr = 'SR2(torch.tensor([0.016, -0.008, -0.008, 0.0, 0.0, 0.0], dtype=torch.float64).reshape(1, 6) * torch.linspace(0.0, 1.0, 1000, dtype=torch.float64).reshape(1000, 1))'
  []
  [p]
    type = Python
    expr = 'Scalar(2.0)'
  []
  [GcbylbyCo]
    type = Python
    expr = 'Scalar(0.0152)'
  []
[]

[Models]
  [degrade]
    type = PowerDegradationFunction
    phase = 'd'
    degradation = 'g'
    power = 'p'
  []
  [sed0]
    type = LinearIsotropicStrainEnergyDensity
    strain = 'E'
    active_strain_energy_density = 'psie_active'
    inactive_strain_energy_density = 'psie_inactive'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    coefficients = '25.84e3 0.18'
    decomposition = 'VOLDEV'
  []
  [sed1]
    type = ScalarMultiplication
    from = 'g psie_active'
    to = 'psie_degraded'
  []
  [sed]
    type = ScalarLinearCombination
    from = 'psie_degraded psie_inactive'
    to = 'psie'
    weights = '1 1'
  []
  [cracked]
    type = CrackGeometricFunctionAT2
    phase = 'd'
    crack = 'alpha'
  []
  [sum]
    type = ScalarLinearCombination
    from = 'alpha psie'
    to = 'psi'
    weights = 'GcbylbyCo 1'
  []
  [energy]
    type = ComposedModel
    models = 'degrade sed0 sed1 sed cracked sum'
  []
  [dpsidd]
    type = Normality
    model = 'energy'
    function = 'psi'
    from = 'd'
    to = 'dpsi_dd'
  []
  [drate]
    type = ScalarVariableRate
    variable = 'd'
  []
  [functional]
    type = ScalarLinearCombination
    from = 'dpsi_dd d_rate'
    to = 'F'
    weights = '1 1'
  []
  [Fish_Burm]
    type = FBComplementarity
    a = 'F'
    b = 'd_rate'
    complementarity = 'd_residual'
    a_inequality = 'LE'
  []
  [eq]
    type = ComposedModel
    models = 'Fish_Burm functional drate dpsidd'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'eq'
    unknowns = 'd'
  []
[]

[Solvers]
  [newton]
    type = Newton
    rel_tol = 1e-08
    abs_tol = 1e-10
    max_its = 50
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [predictor]
    type = LinearExtrapolationPredictor
    unknowns_Scalar = 'd'
  []
  [solve_d]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [stress]
    type = Normality
    model = 'energy'
    function = 'psi'
    from = 'E'
    to = 'S'
  []
  [model]
    type = ComposedModel
    models = 'solve_d stress'
    additional_outputs = 'd'
  []
[]
