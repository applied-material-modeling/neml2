[Tensors]
  [times]
    type = ScalarVTestTimeSeries
    vtest = 'cp_taylor.vtest'
    variable = 'time'
  []
  [deformation_rate]
    type = SR2VTestTimeSeries
    vtest = 'cp_taylor.vtest'
    variable = 'deformation_rate'
  []
  [stresses]
    type = SR2VTestTimeSeries
    vtest = 'cp_taylor.vtest'
    variable = 'stress'
  []
  [vorticity]
    type = WR2VTestTimeSeries
    vtest = 'cp_taylor.vtest'
    variable = 'vorticity'
  []
  [sdirs]
    type = MillerIndex
    values = '1 1 0'
  []
  [splanes]
    type = MillerIndex
    values = '1 1 1'
  []
  [initial_orientation]
    type = Orientation
    input_type = 'euler_angles'
    angle_convention = 'kocks'
    angle_type = 'degrees'
    values = '45 50 51
              75 50 10
              10 5  60
              17 18 19
              30 60 90'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    postprocessor = 'pp'
    prescribed_time = 'times'
    force_SR2_names = 'deformation_rate'
    force_SR2_values = 'deformation_rate'
    force_WR2_names = 'vorticity'
    force_WR2_values = 'vorticity'
    ic_Rot_names = 'orientation'
    ic_Rot_values = 'initial_orientation'
    save_as = 'result.pt'
  []
  [verification]
    type = VTestVerification
    driver = 'driver'
    SR2_names = 'output.mean_cauchy_stress'
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
    lattice_parameter = 1
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
    models = "euler_rodrigues elasticity orientation_rate resolved_shear
              elastic_stretch plastic_deformation_rate plastic_spin
              sum_slip_rates slip_rule slip_strength voce_hardening
              integrate_slip_hardening integrate_elastic_strain integrate_orientation"
  []
[]

[Solvers]
  [newton]
    type = NewtonWithLineSearch
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'elastic_strain slip_hardening orientation'
  []
[]

[Models]
  [cp_warmup_1]
    type = CrystalPlasticityStrainPredictor
    scale = 1
  []
  [cp_warmup_2]
    type = ConstantExtrapolationPredictor
    unknowns_Rot = 'orientation'
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
[]

# Postprocessing
[Models]
  [average_stress]
    type = SR2DynamicMean
    from = 'cauchy_stress'
    to = 'mean_cauchy_stress'
    dim = 0
  []
  [pp]
    type = ComposedModel
    models = 'elasticity average_stress'
  []
[]
