[Tensors]
  [times]
    type = LinspaceScalar
    start = 0
    end = 100
    nstep = 10
    shape_manipulations = 'dynamic_unsqueeze'
    shape_manipulation_args = '(-1)'
  []
  [max_strain]
    type = FillSR2
    values = '0.1 -0.05 -0.05'
  []
  [strains]
    type = LinspaceSR2
    start = 0
    end = max_strain
    nstep = 10
    shape_manipulations = 'dynamic_unsqueeze'
    shape_manipulation_args = '(-1)'
  []
  [E_times]
    type = LinspaceScalar
    start = 0
    end = 30 # we only define the abscissa up to 30, so eventually the ScalarLinearInterpolation will fall out of bound
    nstep = 100
    dim = 0
    group = 'intermediate'
  []
  [E_values]
    type = LinspaceScalar
    start = 1.9e5
    end = 1.2e5
    nstep = 100
    dim = 0
    group = 'intermediate'
  []
[]

[Drivers]
  [driver]
    type = SDTSolidMechanicsDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_strain = 'strains'
  []
[]

[Models]
  [mandel_stress]
    type = IsotropicMandelStress
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/M'
    invariant = 'state/internal/s'
  []
  [yield]
    type = YieldFunction
    yield_stress = 5
  []
  [flow]
    type = ComposedModel
    models = 'vonmises yield'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'state/internal/fp'
    from = 'state/internal/M'
    to = 'state/internal/NM'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 2
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [Erate]
    type = SR2VariableRate
    variable = 'forces/E'
    rate = 'forces/E_rate'
  []
  [Eerate]
    type = SR2LinearCombination
    from_var = 'forces/E_rate state/internal/Ep_rate'
    to_var = 'state/internal/Ee_rate'
    coefficients = '1 -1'
  []
  [youngs_modulus]
    type = ScalarLinearInterpolation
    argument = 'forces/t'
    abscissa = 'E_times'
    ordinate = 'E_values'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = 'youngs_modulus 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    rate_form = true
  []
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/S'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'mandel_stress vonmises yield normality flow_rate Eprate Erate Eerate elasticity integrate_stress'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
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
