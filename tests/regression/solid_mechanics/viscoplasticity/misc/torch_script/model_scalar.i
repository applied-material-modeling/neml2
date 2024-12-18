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
  [G0]
    type = Scalar
    values = '0.0'
    batch_shape = '(${nbatch})'
  []
  [C0]
    type = Scalar
    values = '0.0'
    batch_shape = '(${nbatch})'
  []
[]

[Drivers]
  [driver]
    type = SDTSolidMechanicsDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_strain = 'strains'
    prescribed_temperature = 'temperatures'
    ic_Scalar_names = 'state/G state/C'
    ic_Scalar_values = 'G0 C0'
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
    type = SR2LinearCombination
    to_var = 'forces/Ee'
    from_var = 'forces/E old_state/Ep'
    coefficients = '1 -1'
  []
  [trial_cauchy_stress]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'forces/Ee'
    stress = 'forces/S'
  []
  [trial_flow_direction]
    type = AssociativeJ2FlowDirection
    mandel_stress = 'forces/S'
    flow_direction = 'forces/N'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'forces/S'
    invariant = 'forces/s'
  []
  [trial_state]
    type = ComposedModel
    models = 'trial_elastic_strain trial_cauchy_stress trial_flow_direction vonmises'
  []

  #####################################################################################
  # Stress update
  #####################################################################################
  [ep_rate]
    type = ScalarVariableRate
    variable = 'state/ep'
    rate = 'state/ep_rate'
  []
  [plastic_strain_rate]
    type = AssociativePlasticFlow
    flow_direction = 'forces/N'
    flow_rate = 'state/ep_rate'
    plastic_strain_rate = 'state/Ep_rate'
  []
  [plastic_strain]
    type = SR2ForwardEulerTimeIntegration
    variable = 'state/Ep'
  []
  [plastic_update]
    type = ComposedModel
    models = 'ep_rate plastic_strain_rate plastic_strain'
  []
  [elastic_strain]
    type = SR2LinearCombination
    from_var = 'forces/E state/Ep'
    to_var = 'state/Ee'
    coefficients = '1 -1'
  []
  [cauchy_stress]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'state/Ee'
    stress = 'state/S'
  []
  [stress_update]
    type = ComposedModel
    models = 'elastic_strain cauchy_stress'
  []

  #####################################################################################
  # Compute the rates of equivalent plastic strain and internal variables
  #####################################################################################
  [trial_stress_update]
    type = LinearIsotropicElasticJ2TrialStressUpdate
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    elastic_trial_stress = 'forces/s'
    equivalent_plastic_strain = 'state/ep'
    updated_trial_stress = 'state/s'
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
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/ep'
  []
  [integrate_G]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/G'
  []
  [integrate_C]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/C'
  []
  [rate]
    type = ComposedModel
    models = "trial_stress_update rom
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
    models = 'trial_state radial_return plastic_update stress_update trial_stress_update rom'
    additional_outputs = 'state/s state/ep_rate state/G_rate state/C_rate state/ep state/G state/C state/S state/Ep'
  []
[]
