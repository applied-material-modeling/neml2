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
    value = -0.03
  []
  [ezz]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.01
  []
  [eyz]
    type = FullScalar
    batch_shape = '(20)'
    value = -0.01
  []
  [exz]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.02
  []
  [exy]
    type = FullScalar
    batch_shape = '(20)'
    value = 0.015
  []
  [max_strain]
    type = FillSR2
    values = 'exx eyy ezz eyz exz exy'
  []
  [strains]
    type = LinspaceSR2
    start = 0
    end = max_strain
    nstep = 100
  []
  [f0]
    type = FullScalar
    value = '0.01'
    batch_shape = '(20)'
  []
[]

[Drivers]
  [driver]
    type = SDTSolidMechanicsDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_strain = 'strains'
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

[Models]
  [isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 100
    saturation_rate = 1.2
  []
  [elastic_strain]
    type = SR2LinearCombination
    from_var = 'forces/E state/internal/Ep'
    to_var = 'state/internal/Ee'
    coefficients = '1 -1'
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
  [i1]
    type = SR2Invariant
    invariant_type = 'I1'
    tensor = 'state/internal/M'
    invariant = 'state/internal/sp'
  []
  [yield]
    type = GTNYieldFunction
    yield_stress = 60.0
    q1 = 1.25
    q2 = 1.0
    q3 = 1.57
    isotropic_hardening = 'state/internal/k'
  []
  [flow]
    type = ComposedModel
    models = 'j2 i1 yield'
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
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/internal/Ep'
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/internal/ep'
  []
  [consistency]
    type = RateIndependentPlasticFlowConstraint
  []
  [voidrate]
    type = GursonCavitation
  []
  [integrate_voidrate]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/internal/f'
  []
  [surface]
    type = ComposedModel
    models = "isoharden elastic_strain elasticity
              mandel_stress j2 i1
              yield normality Eprate voidrate
              consistency integrate_Ep integrate_voidrate eprate integrate_ep"
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'surface'
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
    additional_outputs = 'state/internal/Ep'
  []
[]
