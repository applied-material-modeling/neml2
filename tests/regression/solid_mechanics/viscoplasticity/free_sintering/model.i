nbatch = 10
nstep = 100

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
    nstep = ${nstep}
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
    nstep = ${nstep}
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
    nstep = ${nstep}
  []
  [f0]
    type = FullScalar
    value = '0.36'
    batch_shape = '(${nbatch})'
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
    type = SDTSolidMechanicsDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_strain = 'strains'
    prescribed_temperature = 'temperatures'
    ic_Scalar_names = 'state/internal/f'
    ic_Scalar_values = 'f0'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
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
    to_var = 'state/internal/Ee'
    from_var = 'forces/E state/internal/Ep forces/Eg'
    coefficients = '1 -1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '3e4 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
  [mandel_stress]
    type = IsotropicMandelStress
  []
  [j2]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/M'
    invariant = 'state/internal/se'
  []
  [sh]
    type = SR2Invariant
    invariant_type = 'I1'
    tensor = 'state/internal/M'
    invariant = 'state/internal/sh'
  []
  [sp]
    type = ScalarLinearCombination
    to_var = 'state/internal/sp'
    from_var = 'state/internal/sh state/internal/ss'
    coefficients = '1 -1'
  []
  [q1]
    type = ArrheniusParameter
    temperature = 'forces/T'
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
    isotropic_hardening = 'state/internal/k'
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
    function = 'state/internal/fp'
    from = 'state/internal/M state/internal/k'
    to = 'state/internal/NM state/internal/Nk'
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
    variable = 'state/internal/Ep'
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/internal/ep'
  []
  [integrate_void]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/internal/f'
  []
  [surface]
    type = ComposedModel
    models = "isoharden sintering_stress elastic_strain elasticity
              mandel_stress flow flow_rate
              normality
              Eprate eprate voidrate
              integrate_Ep integrate_ep integrate_void"
  []
  [return_map]
    type = ImplicitUpdate
    implicit_model = 'surface'
    solver = 'newton'
  []
  [model]
    type = ComposedModel
    models = 'eigenstrain return_map elastic_strain elasticity'
    additional_outputs = 'state/internal/Ep state/internal/ep state/internal/f'
  []
[]
