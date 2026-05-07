# Wiechert (generalized Maxwell, two branches) response built from primitives — equilibrium spring
# in parallel with two independent Maxwell branches. Same load history and parameters as
# `wiechert/model.i`. Each Maxwell branch carries its own viscous strain (independent internal
# state), and the LinearDashpot output for branch i is renamed to `viscous_strain_i_rate` to align
# with the integrator on `viscous_strain_i`. The N=2 case here generalizes trivially to any N.
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

  # Maxwell branch 1
  [elastic_strain_1]
    type = SR2LinearCombination
    from = 'strain viscous_strain_1'
    weights = '1 -1'
    to = 'elastic_strain_1'
  []
  [maxwell_spring_1]
    type = SR2LinearCombination
    from = 'elastic_strain_1'
    weights = '5000'
    to = 'maxwell_stress_1'
  []
  [dashpot_1]
    type = LinearDashpot
    stress = 'maxwell_stress_1'
    viscous_strain_rate = 'viscous_strain_1_rate'
    viscosity = 100
  []

  # Maxwell branch 2
  [elastic_strain_2]
    type = SR2LinearCombination
    from = 'strain viscous_strain_2'
    weights = '1 -1'
    to = 'elastic_strain_2'
  []
  [maxwell_spring_2]
    type = SR2LinearCombination
    from = 'elastic_strain_2'
    weights = '2000'
    to = 'maxwell_stress_2'
  []
  [dashpot_2]
    type = LinearDashpot
    stress = 'maxwell_stress_2'
    viscous_strain_rate = 'viscous_strain_2_rate'
    viscosity = 1000
  []

  [stress_sum]
    type = SR2LinearCombination
    from = 'eq_stress maxwell_stress_1 maxwell_stress_2'
    weights = '1 1 1'
    to = 'stress'
  []
  [integrate_Ev1]
    type = SR2BackwardEulerTimeIntegration
    variable = 'viscous_strain_1'
  []
  [integrate_Ev2]
    type = SR2BackwardEulerTimeIntegration
    variable = 'viscous_strain_2'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'eq_spring
              elastic_strain_1 maxwell_spring_1 dashpot_1
              elastic_strain_2 maxwell_spring_2 dashpot_2
              stress_sum integrate_Ev1 integrate_Ev2'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'viscous_strain_1 viscous_strain_2'
    residuals = 'viscous_strain_1_residual viscous_strain_2_residual'
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
    unknowns_SR2 = 'viscous_strain_1 viscous_strain_2'
  []
  [update]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model]
    type = ComposedModel
    models = 'update
              eq_spring
              elastic_strain_1 maxwell_spring_1 dashpot_1
              elastic_strain_2 maxwell_spring_2 dashpot_2
              stress_sum'
    additional_outputs = 'viscous_strain_1 viscous_strain_2'
  []
[]
