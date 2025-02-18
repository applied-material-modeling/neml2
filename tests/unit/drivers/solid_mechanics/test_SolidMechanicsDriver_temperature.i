[Tensors]
  [end_time]
    type = LogspaceScalar
    start = -1
    end = 5
    nstep = 20
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = end_time
    nstep = 10
  []
  [exx]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.1
  []
  [eyy]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.05
  []
  [ezz]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.05
  []
  [max_strain]
    type = FillSR2
    values = 'exx eyy ezz'
  []
  [strains]
    type = LinspaceSR2
    start = 0
    end = max_strain
    nstep = 10
  []
  [end_temperature]
    type = LinspaceScalar
    start = 500
    end = 1000
    nstep = 20
  []
  [temperatures]
    type = LinspaceScalar
    start = 400
    end = end_temperature
    nstep = 10
  []
[]

[Drivers]
  [driver]
    type = SDTSolidMechanicsDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_strain = 'strains'
    prescribed_temperature = 'temperatures'
    save_as = 'drivers/solid_mechanics/test_SolidMechanicsDriver_temperature.pt'
  []
[]

[Models]
  [force_rate]
    type = SR2VariableRate
    variable = 'forces/E'
    rate = 'forces/E_rate'
  []
  [youngs_modulus_T]
    type = ArrheniusParameter
    temperature = 'forces/T'
    reference_value = 1e5
    activation_energy = 1e3
    ideal_gas_constant = 8.314
  []
  [stress_rate]
    type = LinearIsotropicElasticity
    coefficients = 'youngs_modulus_T 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    rate_form = true
    strain = 'forces/E'
  []
  [integrate]
    type = SR2ForwardEulerTimeIntegration
    variable = 'state/S'
  []
  [model]
    type = ComposedModel
    models = 'force_rate stress_rate integrate'
  []
[]
