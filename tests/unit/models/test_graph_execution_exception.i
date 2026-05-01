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
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'E'
    force_SR2_values = 'strains'
  []
[]

[Models]
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'stress'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'mandel_stress'
    invariant = 'effective_stress'
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
    function = 'yield_function'
    from = 'mandel_stress'
    to = 'flow_direction'
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
    variable = 'E'
    time = 't'
  []
  [Eerate]
    type = SR2LinearCombination
    from = 'E_rate plastic_strain_rate'
    to = 'strain_rate'
    weights = '1 -1'
  []
  [youngs_modulus]
    type = ScalarLinearInterpolation
    argument = 't'
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
    variable = 'stress'
    time = 't'
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
    unknowns = 'stress'
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
    unknowns_SR2 = 'stress'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
[]
