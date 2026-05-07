# Zener (Standard Linear Solid) response built from primitives — equilibrium spring in parallel
# with a Maxwell branch (spring + dashpot in series). Same load history and parameters as
# `zener/model.i`; gold reference matches that scenario's gold within Newton-solve tolerance.
# Composition pattern:
#   - eq_spring: scalar-modulus spring on the total strain (parallel branch)
#   - elastic_strain = strain - viscous_strain (Maxwell-branch strain decomposition)
#   - maxwell_spring: scalar-modulus spring on the elastic strain
#   - dashpot: LinearDashpot consumes maxwell_stress and produces viscous_strain_rate
#   - stress_sum: parallel branch sum (eq_stress + maxwell_stress)
#   - SR2BackwardEulerTimeIntegration on viscous_strain closes the implicit system
[Tensors]
  [end_time]
    type = LogspaceScalar
    start = -1
    end = 3
    nstep = 10
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = end_time
    nstep = 100
  []
  [exx]
    type = FullScalar
    batch_shape = '(10)'
    value = 0.01
  []
  [eyy]
    type = FullScalar
    batch_shape = '(10)'
    value = -0.005
  []
  [ezz]
    type = FullScalar
    batch_shape = '(10)'
    value = -0.005
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
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Models]
  [eq_spring]
    type = SR2LinearCombination
    from = 'strain'
    weights = '1000'
    to = 'eq_stress'
  []
  [elastic_strain]
    type = SR2LinearCombination
    from = 'strain viscous_strain'
    weights = '1 -1'
    to = 'elastic_strain'
  []
  [maxwell_spring]
    type = SR2LinearCombination
    from = 'elastic_strain'
    weights = '5000'
    to = 'maxwell_stress'
  []
  [dashpot]
    type = LinearDashpot
    stress = 'maxwell_stress'
    viscosity = 100
  []
  [stress_sum]
    type = SR2LinearCombination
    from = 'eq_stress maxwell_stress'
    weights = '1 1'
    to = 'stress'
  []
  [integrate_Ev]
    type = SR2BackwardEulerTimeIntegration
    variable = 'viscous_strain'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'eq_spring elastic_strain maxwell_spring dashpot stress_sum integrate_Ev'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'viscous_strain'
    residuals = 'viscous_strain_residual'
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
    unknowns_SR2 = 'viscous_strain'
  []
  [update]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model]
    type = ComposedModel
    models = 'update eq_spring elastic_strain maxwell_spring dashpot stress_sum'
    additional_outputs = 'viscous_strain'
  []
[]
