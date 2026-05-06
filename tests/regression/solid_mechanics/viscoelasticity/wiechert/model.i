# Stress relaxation of a Wiechert (generalized Maxwell) element with two branches under a ramped
# strain. Each Maxwell branch contributes its own relaxation time scale, and the equilibrium spring
# carries the long-time stress.
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
  [wiechert]
    type = WiechertViscoelasticity
    equilibrium_modulus = 1000
    modulus_1 = 5000
    viscosity_1 = 100
    modulus_2 = 2000
    viscosity_2 = 1000
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
    models = 'wiechert integrate_Ev1 integrate_Ev2'
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
    models = 'update wiechert'
    additional_outputs = 'viscous_strain_1 viscous_strain_2'
  []
[]
