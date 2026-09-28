# neml2
#
# Two yield surfaces on the same Mandel stress: a rate-independent plastic
# surface (Voce isotropic + linear kinematic hardening, closed by a min-map
# complementarity condition) and a power-law creep surface with zero yield
# stress, so it is always active. Each surface carries its own yield function,
# its own Normality operator -- surface 1 is normal to the backstress-shifted
# overstress, surface 2 to the raw Mandel stress, so the two flow directions
# genuinely differ -- and its own consistency parameter. The two inelastic
# strain rates are summed into one inelastic strain, integrated in time.
#
# The load history sweeps end times over six decades, so one model shows both
# limits: at fast rates the rate-independent surface caps the stress and
# carries all the inelastic strain; at slow rates creep carries it instead;
# in between the two surfaces are active simultaneously.

[Tensors]
  [end_time]
    type = Python
    expr = 'Scalar(torch.logspace(-1.0, 5.0, 20, dtype=torch.float64))'
  []
  [times]
    type = Python
    expr = 'Scalar(end_time.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1))'
  []
  [max_strain]
    type = Python
    expr = 'SR2.fill(0.1, -0.05, -0.05, 0.0, 0.0, 0.0).dynamic_batch.expand(20)'
  []
  [strains]
    type = Python
    expr = 'SR2(max_strain.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).reshape(100, 1, 1))'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_SR2_names = 'E'
    prescribed_SR2_values = 'strains'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Models]
  # Kinematics and elasticity: E = Ee + Ein, with Ein = Ep + Ec
  [elastic_strain]
    type = SR2LinearCombination
    from = 'E inelastic_strain'
    to = 'elastic_strain'
    weights = '1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
  []
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'stress'
  []

  # Surface 1: rate-independent plasticity on the overstress
  [isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 100
    saturation_rate = 50
  []
  [kinharden]
    type = LinearKinematicHardening
    hardening_modulus = 500
    back_stress = 'X'
  []
  [overstress]
    type = SR2LinearCombination
    from = 'mandel_stress X'
    to = 'O'
    weights = '1 -1'
  []
  [vonmises_p]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'O'
    invariant = 'effective_overstress'
  []
  [yield_surface_p]
    type = YieldFunction
    yield_stress = 300
    effective_stress = 'effective_overstress'
    isotropic_hardening = 'isotropic_hardening'
    yield_function = 'yield_function_p'
  []
  [flow_p]
    type = ComposedModel
    models = 'overstress vonmises_p yield_surface_p'
  []
  [normality_p]
    type = Normality
    model = 'flow_p'
    function = 'yield_function_p'
    from = 'mandel_stress X isotropic_hardening'
    to = 'flow_direction_p kinematic_hardening_direction isotropic_hardening_direction'
  []
  [consistency_p]
    type = MinMapComplementarity
    a = 'yield_function_p'
    a_inequality = 'LE'
    b = 'flow_rate_p'
  []
  [ep_rate]
    type = AssociativeIsotropicPlasticHardening
    flow_rate = 'flow_rate_p'
  []
  [Kp_rate]
    type = AssociativeKinematicPlasticHardening
    flow_rate = 'flow_rate_p'
  []
  [Ep_rate]
    type = AssociativePlasticFlow
    flow_rate = 'flow_rate_p'
    flow_direction = 'flow_direction_p'
    plastic_strain_rate = 'plastic_strain_rate'
  []

  # Surface 2: power-law creep on the Mandel stress
  [vonmises_c]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'mandel_stress'
    invariant = 'effective_stress'
  []
  [yield_surface_c]
    type = YieldFunction
    yield_stress = 0
    effective_stress = 'effective_stress'
    yield_function = 'yield_function_c'
  []
  [flow_c]
    type = ComposedModel
    models = 'vonmises_c yield_surface_c'
  []
  [normality_c]
    type = Normality
    model = 'flow_c'
    function = 'yield_function_c'
    from = 'mandel_stress'
    to = 'flow_direction_c'
  []
  [creep_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 5000
    exponent = 4
    yield_function = 'yield_function_c'
    flow_rate = 'flow_rate_c'
  []
  [Ec_rate]
    type = AssociativePlasticFlow
    flow_rate = 'flow_rate_c'
    flow_direction = 'flow_direction_c'
    plastic_strain_rate = 'creep_strain_rate'
  []

  # Sum the two mechanisms, then integrate
  [Ein_rate]
    type = SR2LinearCombination
    from = 'plastic_strain_rate creep_strain_rate'
    to = 'inelastic_strain_rate'
    weights = '1 1'
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
  []
  [integrate_Kp]
    type = SR2BackwardEulerTimeIntegration
    variable = 'kinematic_plastic_strain'
  []
  [integrate_Ein]
    type = SR2BackwardEulerTimeIntegration
    variable = 'inelastic_strain'
  []
  [surface]
    type = ComposedModel
    models = 'elastic_strain elasticity mandel_stress
              isoharden kinharden overstress vonmises_p yield_surface_p
              normality_p consistency_p ep_rate Kp_rate Ep_rate
              vonmises_c yield_surface_c normality_c creep_rate Ec_rate
              Ein_rate integrate_ep integrate_Kp integrate_Ein'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'surface'
    unknowns = 'inelastic_strain kinematic_plastic_strain equivalent_plastic_strain flow_rate_p'
    residuals = 'inelastic_strain_residual kinematic_plastic_strain_residual
                 equivalent_plastic_strain_residual complementarity'
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
    unknowns_SR2 = 'inelastic_strain kinematic_plastic_strain'
    unknowns_Scalar = 'equivalent_plastic_strain flow_rate_p'
  []
  [return_map]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model]
    type = ComposedModel
    models = 'return_map elastic_strain elasticity'
    additional_outputs = 'inelastic_strain kinematic_plastic_strain equivalent_plastic_strain'
  []
[]
