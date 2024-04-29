ntime = 100
nbatch = 20

[Tensors]
  [end_time]
    type = LogspaceScalar
    start = 0
    end = 1
    nstep = ${nbatch}
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = end_time
    nstep = ${ntime}
  []
  [start_temperature]
    type = LinspaceScalar
    start = 300
    end = 500
    nstep = ${nbatch}
  []
  [end_temperature]
    type = LinspaceScalar
    start = 1800
    end = 1200
    nstep = ${nbatch}
  []
  [temperatures]
    type = LinspaceScalar
    start = start_temperature
    end = end_temperature
    nstep = ${ntime}
  []
  [exx]
    type = FullScalar
    batch_shape = '(${nbatch})'
    value = 0.1
  []
  [eyy]
    type = FullScalar
    batch_shape = '(${nbatch})'
    value = -0.05
  []
  [ezz]
    type = FullScalar
    batch_shape = '(${nbatch})'
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
    nstep = ${ntime}
  []
[]

[Drivers]
  [driver]
    type = SolidMechanicsDriver
    model = 'model'
    times = 'times'
    prescribed_strains = 'strains'
    prescribed_temperatures = 'temperatures'
    predictor = LINEAR_EXTRAPOLATION
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
    abs_tol = 1e-8
    rel_tol = 1e-6
  []
[]

[Models]
  #####################################################################################
  # Compute the invariant plastic flow direction since we are doing J2 radial return
  #####################################################################################
  [trial_elastic_strain]
    type = ElasticStrain
    plastic_strain = 'old_state/Ep'
    elastic_strain = 'state/Ee'
  []
  [cauchy_stress]
    type = LinearIsotropicElasticity
    youngs_modulus = 1e5
    poisson_ratio = 0.3
    strain = 'state/Ee'
    stress = 'state/S'
  []
  [flow_direction]
    type = J2FlowDirection
    mandel_stress = 'state/S'
    flow_direction = 'forces/N'
  []
  [trial_state]
    type = ComposedModel
    models = 'trial_elastic_strain cauchy_stress flow_direction'
  []

  #####################################################################################
  # Stress update
  #####################################################################################
  [ep_rate]
    type = ScalarStateRate
    state = 'ep'
  []
  [plastic_strain_rate]
    type = AssociativePlasticFlow
    flow_direction = 'forces/N'
    flow_rate = 'state/ep_rate'
    plastic_strain_rate = 'state/Ep_rate'
  []
  [plastic_strain]
    type = SR2ForwardEulerTimeIntegration
    variable = 'Ep'
  []
  [plastic_update]
    type = ComposedModel
    models = 'ep_rate plastic_strain_rate plastic_strain'
  []
  [elastic_strain]
    type = ElasticStrain
    plastic_strain = 'state/Ep'
    elastic_strain = 'state/Ee'
  []
  [stress_update]
    type = ComposedModel
    models = 'elastic_strain cauchy_stress'
  []

  #####################################################################################
  # Compute the rates of equivalent plastic strain and internal variables
  #####################################################################################
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/S'
    invariant = 'state/s'
  []
  [rom]
    type = TorchScriptFlowRate
    von_mises_stress = 'state/s'
    temperature = 'forces/T'
    internal_state_1 = 'state/G'
    internal_state_2 = 'state/C'
    equivalent_plastic_strain_rate = 'state/ep_rate'
    internal_state_1_rate = 'state/G_rate'
    internal_state_2_rate = 'state/C_rate'
    torch_script = 'gold/surrogate.pt'
    use_AD_first_derivative = true
    use_AD_second_derivative = true
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'ep'
  []
  [integrate_G]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'G'
  []
  [integrate_C]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'C'
  []
  [rate]
    type = ComposedModel
    models = "plastic_update stress_update vonmises rom
              integrate_ep integrate_G integrate_C"
  []
  [radial_return]
    type = ImplicitUpdate
    implicit_model = 'rate'
    solver = 'newton'
  []

  #####################################################################################
  # Put the models together
  #####################################################################################
  [model]
    type = ComposedModel
    models = 'trial_state radial_return plastic_update stress_update vonmises rom'
    additional_outputs = 'state/s state/ep state/G state/C state/S state/Ep'
  []
[]
