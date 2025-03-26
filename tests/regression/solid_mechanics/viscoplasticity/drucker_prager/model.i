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
    nstep = 100
  []
  [exx]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.1
  []
  [eyy]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.01
  []
  [ezz]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.02
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
  []
[]

[Models]
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'state/S'
    mandel_stress = 'state/M'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/M'
    invariant = 'state/svm'
  []
  [first_invariant]
    type = SR2Invariant
    invariant_type = 'I1'
    tensor = 'state/M'
    invariant = 'state/i1'
  []
  [effective_stress]
    type = ScalarLinearCombination
    from_var = 'state/svm state/i1'
    to_var = 'state/s'
    coefficients = '1 -0.5' # coefficients for Drucker-Prager, note that von Mises stress is sqrt(3 J2)
  []
  [isoharden]
    type = LinearIsotropicHardening
    equivalent_plastic_strain = 'state/ep'
    isotropic_hardening = 'state/k'
    hardening_modulus = 1000
  []
  [yield]
    type = YieldFunction
    yield_stress = 50
    effective_stress = 'state/s'
    isotropic_hardening = 'state/k'
    yield_function = 'state/fp'
  []
  [flow]
    type = ComposedModel
    models = 'vonmises first_invariant effective_stress yield'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'state/fp'
    from = 'state/M state/k'
    to = 'state/NM state/Nk'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    yield_function = 'state/fp'
    flow_rate = 'state/gamma_rate'
    reference_stress = 100
    exponent = 2
  []
  [Eprate]
    type = AssociativePlasticFlow
    flow_rate = 'state/gamma_rate'
    flow_direction = 'state/NM'
    plastic_strain_rate = 'state/Ep_rate'
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
    flow_rate = 'state/gamma_rate'
    isotropic_hardening_direction = 'state/Nk'
    equivalent_plastic_strain_rate = 'state/ep_rate'
  []
  [Erate]
    type = SR2VariableRate
    variable = 'forces/E'
    rate = 'forces/E_rate'
  []
  [Eerate]
    type = SR2LinearCombination
    from_var = 'forces/E_rate state/Ep_rate'
    to_var = 'state/Ee_rate'
    coefficients = '1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'state/Ee'
    stress = 'state/S'
    rate_form = true
  []
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/S'
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/ep'
  []
  [implicit_rate]
    type = ComposedModel
    models = "mandel_stress isoharden flow normality
              flow_rate
              Eprate eprate Erate Eerate elasticity
              integrate_stress integrate_ep"
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
[]
