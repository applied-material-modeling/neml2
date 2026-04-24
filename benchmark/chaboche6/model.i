[Tensors]
  [end_time]
    type = LogspaceScalar
    start = 5
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
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'strain'
    force_SR2_values = 'strains'
    device = ${device}
  []
[]

[Models]
  [isoharden]
    type = VoceIsotropicHardening
    saturated_hardening = 100
    saturation_rate = 1.2
  []
  [kinharden]
    type = SR2LinearCombination
    from = 'X1 X2 X3 X4 X5 X6'
    to = 'back_stress'
  []
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'stress'
  []
  [overstress]
    type = SR2LinearCombination
    from = 'mandel_stress back_stress'
    to = 'overstress'
    weights = '1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'overstress'
    invariant = 'effective_stress'
  []
  [yield]
    type = YieldFunction
    yield_stress = 5
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
    from = 'mandel_stress isotropic_hardening'
    to = 'flow_direction isotropic_hardening_direction'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 2
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [X1rate]
    type = ChabochePlasticHardening
    back_stress = 'X1'
    C = 10000
    g = 100
    A = 1e-8
    a = 1.2
  []
  [X2rate]
    type = ChabochePlasticHardening
    back_stress = 'X2'
    C = 1000
    g = 9
    A = 1e-10
    a = 3.2
  []
  [X3rate]
    type = ChabochePlasticHardening
    back_stress = 'X3'
    C = 10000
    g = 100
    A = 1e-8
    a = 1.2
  []
  [X4rate]
    type = ChabochePlasticHardening
    back_stress = 'X4'
    C = 1000
    g = 9
    A = 1e-10
    a = 3.2
  []
  [X5rate]
    type = ChabochePlasticHardening
    back_stress = 'X5'
    C = 10000
    g = 100
    A = 1e-8
    a = 1.2
  []
  [X6rate]
    type = ChabochePlasticHardening
    back_stress = 'X6'
    C = 1000
    g = 9
    A = 1e-10
    a = 3.2
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [Erate]
    type = SR2VariableRate
    variable = 'strain'
  []
  [Eerate]
    type = SR2LinearCombination
    from = 'strain_rate plastic_strain_rate'
    to = 'elastic_strain_rate'
    weights = '1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
    rate_form = true
  []
  [integrate_ep]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'equivalent_plastic_strain'
  []
  [integrate_X1]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X1'
  []
  [integrate_X2]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X2'
  []
  [integrate_X3]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X3'
  []
  [integrate_X4]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X4'
  []
  [integrate_X5]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X5'
  []
  [integrate_X6]
    type = SR2BackwardEulerTimeIntegration
    variable = 'X6'
  []
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'stress'
  []
  [implicit_rate]
    type = ComposedModel
    models = "isoharden kinharden mandel_stress overstress vonmises yield normality
              flow_rate eprate Eprate X1rate X2rate X3rate X4rate X5rate X6rate Erate Eerate elasticity
              integrate_stress integrate_ep integrate_X1 integrate_X2 integrate_X3 integrate_X4 integrate_X5 integrate_X6"
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'stress equivalent_plastic_strain X1 X2 X3 X4 X5 X6'
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
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
[]
