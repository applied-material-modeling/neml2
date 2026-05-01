[Tensors]
  [end_time]
    type = LogspaceScalar
    start = 3
    end = 3
    nstep = ${nbatch}
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = end_time
    nstep = 100
  []
  [start_temperature]
    type = LinspaceScalar
    start = 300
    end = 300
    nstep = ${nbatch}
  []
  [end_temperature]
    type = LinspaceScalar
    start = 1800
    end = 1800
    nstep = ${nbatch}
  []
  [temperatures]
    type = LinspaceScalar
    start = start_temperature
    end = end_temperature
    nstep = 100
  []
  [exx]
    type = FullScalar
    batch_shape = '(${nbatch})'
    value = 0
  []
  [eyy]
    type = FullScalar
    batch_shape = '(${nbatch})'
    value = 0
  []
  [ezz]
    type = FullScalar
    batch_shape = '(${nbatch})'
    value = 0
  []
  [max_strain]
    type = FillSR2
    values = 'exx eyy ezz'
  []
  [strains]
    type = LinspaceSR2
    start = 0
    end = max_strain
    nstep = 100
  []
  [f0]
    type = Scalar
    values = '0.36'
  []
  [gamma]
    type = LinspaceScalar
    start = 0
    end = 150
    nstep = ${nbatch}
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'strain'
    force_SR2_values = 'strains'
    force_Scalar_names = 'temperature'
    force_Scalar_values = 'temperatures'
    ic_Scalar_names = 'void_fraction'
    ic_Scalar_values = 'f0'
    device = ${device}
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
    from = 'strain plastic_strain eigenstrain'
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
    models = "isoharden sintering_stress elastic_strain elasticity
              mandel_stress flow flow_rate
              normality
              Eprate eprate voidrate
              integrate_Ep integrate_ep integrate_void"
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'surface'
    unknowns = 'plastic_strain equivalent_plastic_strain void_fraction'
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
