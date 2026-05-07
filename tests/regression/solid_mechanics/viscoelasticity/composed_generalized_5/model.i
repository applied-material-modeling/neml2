# Generalized 5-element Maxwell network: equilibrium spring in parallel with FOUR Maxwell branches,
# each with its own relaxation time scale (Prony-series-style). No named element model exists for
# this topology — this scenario demonstrates the composition pattern's reach beyond the named cases.
# Adding an Nth branch is the same recipe N times: elastic_strain_i, maxwell_spring_i, dashpot_i,
# integrate_Ev_i, plus i extra entries in unknowns / residuals / stress_sum.
#
# Branch parameters (pick spread time scales for an interesting relaxation curve):
#   eq_modulus = 1000
#   branch 1: E = 5000, eta = 100   (tau = 0.02)
#   branch 2: E = 3000, eta = 300   (tau = 0.1)
#   branch 3: E = 2000, eta = 1000  (tau = 0.5)
#   branch 4: E = 1000, eta = 5000  (tau = 5)
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

  # Branch 1
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
  [integrate_Ev1]
    type = SR2BackwardEulerTimeIntegration
    variable = 'viscous_strain_1'
  []

  # Branch 2
  [elastic_strain_2]
    type = SR2LinearCombination
    from = 'strain viscous_strain_2'
    weights = '1 -1'
    to = 'elastic_strain_2'
  []
  [maxwell_spring_2]
    type = SR2LinearCombination
    from = 'elastic_strain_2'
    weights = '3000'
    to = 'maxwell_stress_2'
  []
  [dashpot_2]
    type = LinearDashpot
    stress = 'maxwell_stress_2'
    viscous_strain_rate = 'viscous_strain_2_rate'
    viscosity = 300
  []
  [integrate_Ev2]
    type = SR2BackwardEulerTimeIntegration
    variable = 'viscous_strain_2'
  []

  # Branch 3
  [elastic_strain_3]
    type = SR2LinearCombination
    from = 'strain viscous_strain_3'
    weights = '1 -1'
    to = 'elastic_strain_3'
  []
  [maxwell_spring_3]
    type = SR2LinearCombination
    from = 'elastic_strain_3'
    weights = '2000'
    to = 'maxwell_stress_3'
  []
  [dashpot_3]
    type = LinearDashpot
    stress = 'maxwell_stress_3'
    viscous_strain_rate = 'viscous_strain_3_rate'
    viscosity = 1000
  []
  [integrate_Ev3]
    type = SR2BackwardEulerTimeIntegration
    variable = 'viscous_strain_3'
  []

  # Branch 4
  [elastic_strain_4]
    type = SR2LinearCombination
    from = 'strain viscous_strain_4'
    weights = '1 -1'
    to = 'elastic_strain_4'
  []
  [maxwell_spring_4]
    type = SR2LinearCombination
    from = 'elastic_strain_4'
    weights = '1000'
    to = 'maxwell_stress_4'
  []
  [dashpot_4]
    type = LinearDashpot
    stress = 'maxwell_stress_4'
    viscous_strain_rate = 'viscous_strain_4_rate'
    viscosity = 5000
  []
  [integrate_Ev4]
    type = SR2BackwardEulerTimeIntegration
    variable = 'viscous_strain_4'
  []

  [stress_sum]
    type = SR2LinearCombination
    from = 'eq_stress maxwell_stress_1 maxwell_stress_2 maxwell_stress_3 maxwell_stress_4'
    weights = '1 1 1 1 1'
    to = 'stress'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'eq_spring
              elastic_strain_1 maxwell_spring_1 dashpot_1 integrate_Ev1
              elastic_strain_2 maxwell_spring_2 dashpot_2 integrate_Ev2
              elastic_strain_3 maxwell_spring_3 dashpot_3 integrate_Ev3
              elastic_strain_4 maxwell_spring_4 dashpot_4 integrate_Ev4
              stress_sum'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'viscous_strain_1 viscous_strain_2 viscous_strain_3 viscous_strain_4'
    residuals = 'viscous_strain_1_residual viscous_strain_2_residual
                 viscous_strain_3_residual viscous_strain_4_residual'
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
    unknowns_SR2 = 'viscous_strain_1 viscous_strain_2 viscous_strain_3 viscous_strain_4'
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
              elastic_strain_3 maxwell_spring_3 dashpot_3
              elastic_strain_4 maxwell_spring_4 dashpot_4
              stress_sum'
    additional_outputs = 'viscous_strain_1 viscous_strain_2 viscous_strain_3 viscous_strain_4'
  []
[]
