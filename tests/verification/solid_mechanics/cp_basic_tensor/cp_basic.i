# neml2
# Native port of tests/verification/solid_mechanics/cp_basic_tensor/cp_basic.i.
# Same scenario as cp_basic but routed through IsotropicElasticityTensor + GeneralElasticity
# (anisotropic-elasticity machinery on isotropic constants), exercising the tensor pathway.
# Reference data is loaded from cp_basic.csv (converted from the original .vtest).
[Tensors]
  [times]
    type = CSVScalar
    csv_file = 'cp_basic.csv'
    variable = 'time'
  []
  [deformation_rate]
    type = CSVSR2
    csv_file = 'cp_basic.csv'
    variable = 'deformation_rate'
  []
  [stresses]
    type = CSVSR2
    csv_file = 'cp_basic.csv'
    variable = 'stress'
  []
  [vorticity]
    type = CSVWR2
    csv_file = 'cp_basic.csv'
    variable = 'vorticity'
  []
  [sdirs]
    type = Python
    expr = 'MillerIndex(torch.tensor([1.0, 1.0, 0.0]))'
  []
  [splanes]
    type = Python
    expr = 'MillerIndex(torch.tensor([1.0, 1.0, 1.0]))'
  []
  [initial_orientation]
    type = Python
    expr = 'MRP(torch.tensor([-0.54412095, -0.34931944, 0.12600655], dtype=torch.float64))'
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
  [verification]
    type = Verification
    driver = 'driver'
    SR2_names = 'output.cauchy_stress'
    SR2_values = 'stresses'
    # Looser tolerances here are because the NEML(1) model was generated with lagged, explict
    # integration on the orientations
    atol = 1.0
    rtol = 1e-3
  []
[]

[Data]
  [crystal_geometry]
    type = CubicCrystal
    lattice_parameter = '1.0'
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
  [elastic_tensor]
    type = IsotropicElasticityTensor
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    coefficients = '1e5 0.25'
  []
  [elasticity]
    type = GeneralElasticity
    elastic_stiffness_tensor = 'elastic_tensor'
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
    models = 'euler_rodrigues elasticity orientation_rate
              resolved_shear elastic_stretch plastic_deformation_rate
              plastic_spin sum_slip_rates slip_rule slip_strength voce_hardening
              integrate_slip_hardening integrate_elastic_strain integrate_orientation'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'elastic_strain slip_hardening orientation'
  []
[]

[Solvers]
  [newton]
    type = NewtonWithLineSearch
    linesearch_cutback = 2.0
    linesearch_stopping_criteria = 1.0e-3
    max_linesearch_iterations = 5
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [cp_warmup_1]
    type = CrystalPlasticityStrainPredictor
    scale = 0.05
  []
  [cp_warmup_2]
    type = ConstantExtrapolationPredictor
    unknowns_MRP = 'orientation'
    unknowns_Scalar = 'slip_hardening'
  []
  [predictor]
    type = ComposedModel
    models = 'cp_warmup_1 cp_warmup_2'
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
    additional_outputs = 'elastic_strain orientation'
  []
[]
