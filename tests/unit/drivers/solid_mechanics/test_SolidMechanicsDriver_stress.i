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
  [sxx]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.1
  []
  [syy]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.05
  []
  [szz]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.05
  []
  [max_stress]
    type = FillSR2
    values = 'sxx syy szz'
  []
  [stresses]
    type = LinspaceSR2
    start = 0
    end = max_stress
    nstep = 10
  []
[]

[Drivers]
  [driver]
    type = SDTSolidMechanicsDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_stress = 'stresses'
    control = 'STRESS'
    save_as = 'drivers/solid_mechanics/test_SolidMechanicsDriver_stress.pt'
  []
[]

[Models]
  [force_rate]
    type = SR2VariableRate
    variable = 'forces/S'
    rate = 'forces/S_rate'
  []
  [strain_rate]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    rate_form = true
    compliance = true
    stress = 'forces/S'
    strain = 'state/E'
  []
  [integrate]
    type = SR2ForwardEulerTimeIntegration
    variable = 'state/E'
  []
  [model]
    type = ComposedModel
    models = 'force_rate strain_rate integrate'
  []
[]
