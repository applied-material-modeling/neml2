# neml2
# Native port of tests/regression/solid_mechanics/viscoplasticity/free_sintering/model.i.
# Free sintering at zero applied strain with thermal eigenstrain + Olevsky-Skorohod
# sintering stress driving GTN poroplasticity. Batch (100, 10): 100 time steps,
# 10 surface-tension values (gamma in [0, 150]).
[Tensors]
  # end_time = LogspaceScalar(3, 3, 10) -> 10 copies of 10^3 = 1000.
  [end_time]
    type = Python
    expr = 'Scalar(torch.logspace(3.0, 3.0, 10, dtype=torch.float64))'
  []
  # times = LinspaceScalar(0, end_time, 100) -> shape (100, 10).
  [times]
    type = Python
    expr = 'Scalar(end_time.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1))'
  []
  # start_temperature = LinspaceScalar(300, 300, 10) -> 10 copies of 300.
  [start_temperature]
    type = Python
    expr = 'Scalar(torch.linspace(300.0, 300.0, 10, dtype=torch.float64))'
  []
  # end_temperature = LinspaceScalar(1800, 1800, 10) -> 10 copies of 1800.
  [end_temperature]
    type = Python
    expr = 'Scalar(torch.linspace(1800.0, 1800.0, 10, dtype=torch.float64))'
  []
  # temperatures = LinspaceScalar(start_temperature, end_temperature, 100) -> (100, 10).
  [temperatures]
    type = Python
    expr = 'Scalar(start_temperature.data.unsqueeze(0) + (end_temperature.data - start_temperature.data).unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1))'
  []
  # max_strain = FillSR2(exx=0, eyy=0, ezz=0) batched (10,) -> all zeros.
  # Diagonal 3-arg fill places exx/eyy/ezz on slots 0/1/2 with no Mandel scaling.
  [max_strain]
    type = Python
    expr = 'SR2(torch.zeros((10, 6), dtype=torch.float64))'
  []
  # strains = LinspaceSR2(0, max_strain, 100) -> (100, 10, 6), all zeros.
  [strains]
    type = Python
    expr = 'SR2(torch.zeros((100, 10, 6), dtype=torch.float64))'
  []
  # f0 = FullScalar(0.36, (10,))
  [f0]
    type = Python
    expr = 'Scalar(torch.full((10,), 0.36, dtype=torch.float64))'
  []
  # gamma = LinspaceScalar(0, 150, 10) -> shape (10,)
  [gamma]
    type = Python
    expr = 'Scalar(torch.linspace(0.0, 150.0, 10, dtype=torch.float64))'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'E'
    force_SR2_values = 'strains'
    force_Scalar_names = 'temperature'
    force_Scalar_values = 'temperatures'
    ic_Scalar_names = 'void_fraction'
    ic_Scalar_values = 'f0'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Models]
  [isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 5
    saturation_rate = 1.2
  []
  [sintering_stress]
    type = OlevskySinteringStress
    surface_tension = 'gamma'
    particle_radius = 3e-4
  []
  [eigenstrain]
    type = ThermalEigenstrain
    reference_temperature = 300
    CTE = 1e-6
  []
  [elastic_strain]
    type = SR2LinearCombination
    from = 'E plastic_strain eigenstrain'
    to = 'elastic_strain'
    weights = '1 -1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '3e4 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
  []
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'stress'
  []
  [j2]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'mandel_stress'
    invariant = 'flow_invariant'
  []
  [sh]
    type = SR2Invariant
    invariant_type = 'I1'
    tensor = 'mandel_stress'
    invariant = 'hydrostatic_stress'
  []
  [sp]
    type = ScalarLinearCombination
    from = 'hydrostatic_stress sintering_stress'
    to = 'poro_invariant'
    weights = '1 -1'
  []
  [q1]
    type = ArrheniusParameter
    temperature = 'temperature'
    reference_value = 8000
    activation_energy = 5e4
    ideal_gas_constant = 8.314
  []
  [yield]
    type = GTNYieldFunction
    yield_stress = 60.0
    q1 = 'q1'
    q2 = 0.01
    q3 = 1.57
    isotropic_hardening = 'isotropic_hardening'
  []
  [flow]
    type = ComposedModel
    models = 'j2 sh sp yield'
    automatic_nonlinear_parameter = false
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 500
    exponent = 2
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'yield_function'
    from = 'mandel_stress isotropic_hardening'
    to = 'flow_direction isotropic_hardening_direction'
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [voidrate]
    type = GursonCavitation
  []
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'plastic_strain'
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
  []
  [integrate_void]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'void_fraction'
  []
  [surface]
    type = ComposedModel
    models = 'isoharden sintering_stress elastic_strain elasticity mandel_stress flow flow_rate normality Eprate eprate voidrate integrate_Ep integrate_ep integrate_void'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'surface'
    unknowns = 'plastic_strain equivalent_plastic_strain void_fraction'
    residuals = 'plastic_strain_residual equivalent_plastic_strain_residual void_fraction_residual'
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
    type = ConstantExtrapolationPredictor
    unknowns_SR2 = 'plastic_strain'
    unknowns_Scalar = 'equivalent_plastic_strain void_fraction'
  []
  [return_map]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model]
    type = ComposedModel
    models = 'eigenstrain return_map elastic_strain elasticity'
    additional_outputs = 'plastic_strain equivalent_plastic_strain void_fraction'
  []
[]
