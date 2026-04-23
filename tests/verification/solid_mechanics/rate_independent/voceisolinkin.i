[Tensors]
  [times]
    type = ScalarVTestTimeSeries
    vtest = 'voceisolinkin.vtest'
    variable = 'time'
  []
  [strains]
    type = SR2VTestTimeSeries
    vtest = 'voceisolinkin.vtest'
    variable = 'strain'
  []
  [stresses]
    type = SR2VTestTimeSeries
    vtest = 'voceisolinkin.vtest'
    variable = 'stress'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'E'
    force_SR2_values = 'strains'
    save_as = 'result.pt'
  []
  [verification]
    type = VTestVerification
    driver = 'driver'
    SR2_names = 'output.stress'
    SR2_values = 'stresses'
    rtol = 1e-5
    atol = 1e-8
  []
[]

[Models]
  [isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 100
    saturation_rate = 10.0
  []
  [kinharden]
    type = LinearKinematicHardening
    hardening_modulus = 1000
    back_stress = 'X'
  []
  [elastic_strain]
    type = SR2LinearCombination
    from = 'E plastic_strain'
    to = 'elastic_strain'
    weights = '1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '120000 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
  []
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'stress'
  []
  [overstress]
    type = SR2LinearCombination
    to = 'O'
    from = 'mandel_stress X'
    weights = '1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'O'
    invariant = 'effective_stress'
  []
  [yield]
    type = YieldFunction
    yield_stress = 100
    isotropic_hardening = 'isotropic_hardening'
  []
  [flow]
    type = ComposedModel
    models = 'overstress vonmises yield'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'yield_function'
    from = 'mandel_stress X isotropic_hardening'
    to = 'flow_direction kinematic_hardening_direction isotropic_hardening_direction'
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [Kprate]
    type = AssociativeKinematicPlasticHardening
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
  []
  [integrate_Kp]
    type = SR2BackwardEulerTimeIntegration
    variable = 'kinematic_plastic_strain'
  []
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'plastic_strain'
  []
  [consistency]
    type = FischerBurmeister
    a = 'yield_function'
    a_inequality = 'LE'
    b = 'flow_rate'
  []
  [surface]
    type = ComposedModel
    models = "isoharden kinharden elastic_strain elasticity
              mandel_stress overstress vonmises
              yield normality eprate Kprate Eprate
              consistency integrate_ep integrate_Kp integrate_Ep"
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'surface'
    unknowns = 'plastic_strain kinematic_plastic_strain equivalent_plastic_strain flow_rate'
    residuals = 'plastic_strain_residual kinematic_plastic_strain_residual equivalent_plastic_strain_residual complementarity'
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
  [return_map]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
  [model]
    type = ComposedModel
    models = 'return_map elastic_strain elasticity'
    additional_outputs = 'plastic_strain kinematic_plastic_strain equivalent_plastic_strain'
  []
[]
