[Tensors]
  [end_time]
    type = LogspaceScalar
    start = -1
    end = 5
    nstep = ${nbatch}
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = end_time
    nstep = 100
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
    nstep = 100
  []
[]

[Drivers]
  [driver]
    type = SDTSolidMechanicsDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_strain = 'strains'
    predictor = 'LINEAR_EXTRAPOLATION'
    device = ${device}
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
  ###############################################################################
  # Use the trial state to precalculate invariant flow directions
  # prior to radial return
  ###############################################################################
  [trial_elastic_strain]
    type = SR2LinearCombination
    from_var = 'forces/E old_state/internal/Ep'
    to_var = 'forces/Ee_trial'
    coefficients = '1 -1'
  []
  [trial_cauchy_stress]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'forces/Ee_trial'
    stress = 'forces/S_trial'
  []
  [trial_mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'forces/S_trial'
    mandel_stress = 'forces/M_trial'
  []
  [trial_isoharden]
    type = LinearIsotropicHardening
    equivalent_plastic_strain = 'old_state/internal/ep'
    isotropic_hardening = 'forces/k_trial'
    hardening_modulus = 1000
  []
  [trial_kinharden]
    type = LinearKinematicHardening
    kinematic_plastic_strain = 'old_state/internal/Kp'
    back_stress = 'forces/X_trial'
    hardening_modulus = 1000
  []
  [trial_overstress]
    type = SR2LinearCombination
    to_var = 'forces/O_trial'
    from_var = 'forces/M_trial forces/X_trial'
    coefficients = '1 -1'
  []
  [trial_vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'forces/O_trial'
    invariant = 'forces/s_trial'
  []
  [trial_yield]
    type = YieldFunction
    yield_stress = 5
    yield_function = 'forces/fp_trial'
    effective_stress = 'forces/s_trial'
    isotropic_hardening = 'forces/k_trial'
  []
  [trial_flow]
    type = ComposedModel
    models = 'trial_overstress trial_vonmises trial_yield'
  []
  [trial_normality]
    type = Normality
    model = 'trial_flow'
    function = 'forces/fp_trial'
    from = 'forces/M_trial forces/k_trial forces/X_trial'
    to = 'forces/NM forces/Nk forces/NX'
  []
  [trial_state]
    type = ComposedModel
    models = "trial_elastic_strain trial_cauchy_stress trial_mandel_stress
              trial_isoharden trial_kinharden trial_normality"
  []
  ###############################################################################
  # The actual radial return:
  # Since the flow directions are invariant, we only need to integrate
  # the consistency parameter.
  ###############################################################################
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 1000
  []
  [kinharden]
    type = LinearKinematicHardening
    hardening_modulus = 1000
  []
  [trial_flow_rate]
    type = ScalarVariableRate
    variable = 'state/internal/gamma'
  []
  [plastic_strain_rate]
    type = AssociativePlasticFlow
    flow_direction = 'forces/NM'
  []
  [plastic_strain]
    type = SR2ForwardEulerTimeIntegration
    variable = 'state/internal/Ep'
  []
  [elastic_strain]
    type = SR2LinearCombination
    from_var = 'forces/E state/internal/Ep'
    to_var = 'state/internal/Ee'
    coefficients = '1 -1'
  []
  [cauchy_stress]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
  [mandel_stress]
    type = IsotropicMandelStress
  []
  [overstress]
    type = SR2LinearCombination
    to_var = 'state/internal/O'
    from_var = 'state/internal/M state/internal/X'
    coefficients = '1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/O'
    invariant = 'state/internal/s'
  []
  [yield]
    type = YieldFunction
    yield_stress = 5
    isotropic_hardening = 'state/internal/k'
  []
  [equivalent_plastic_strain_rate]
    type = AssociativeIsotropicPlasticHardening
    isotropic_hardening_direction = 'forces/Nk'
  []
  [equivalent_plastic_strain]
    type = ScalarForwardEulerTimeIntegration
    variable = 'state/internal/ep'
  []
  [kinematic_plastic_strain_rate]
    type = AssociativeKinematicPlasticHardening
    kinematic_hardening_direction = 'forces/NX'
  []
  [kinematic_plastic_strain]
    type = SR2ForwardEulerTimeIntegration
    variable = 'state/internal/Kp'
  []
  [surface]
    type = ComposedModel
    models = "trial_flow_rate
              plastic_strain_rate plastic_strain elastic_strain cauchy_stress mandel_stress
              kinematic_plastic_strain_rate kinematic_plastic_strain kinharden
              equivalent_plastic_strain_rate equivalent_plastic_strain isoharden
              overstress vonmises yield"
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 2
  []
  [integrate_gamma]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/internal/gamma'
  []
  [implicit_rate]
    type = ComposedModel
    models = "surface flow_rate integrate_gamma"
  []
  [return_map]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
  [model0]
    type = ComposedModel
    models = "trial_state return_map trial_flow_rate
              plastic_strain_rate plastic_strain
              equivalent_plastic_strain_rate equivalent_plastic_strain
              kinematic_plastic_strain_rate kinematic_plastic_strain"
    additional_outputs = 'state/internal/gamma'
  []
  [model]
    type = ComposedModel
    models = 'model0 elastic_strain cauchy_stress'
    additional_outputs = 'state/internal/Ep state/internal/ep state/internal/Kp'
  []
[]
