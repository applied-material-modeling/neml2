# Creep-like response of a Burgers viscoelastic element under a ramped strain history. The Burgers
# model has two internal strains (Maxwell dashpot viscous strain and Kelvin-Voigt branch strain)
# that are solved for via an implicit Newton update at each time step.
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
  [burgers]
    type = BurgersElement
    maxwell_modulus = 5000
    maxwell_viscosity = 200
    kelvin_modulus = 2000
    kelvin_viscosity = 100
  []
  [integrate_EvM]
    type = SR2BackwardEulerTimeIntegration
    variable = 'maxwell_viscous_strain'
  []
  [integrate_EK]
    type = SR2BackwardEulerTimeIntegration
    variable = 'kelvin_voigt_strain'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'burgers integrate_EvM integrate_EK'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'maxwell_viscous_strain kelvin_voigt_strain'
    residuals = 'maxwell_viscous_strain_residual kelvin_voigt_strain_residual'
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
    unknowns_SR2 = 'maxwell_viscous_strain kelvin_voigt_strain'
  []
  [update]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model]
    type = ComposedModel
    models = 'update burgers'
    additional_outputs = 'maxwell_viscous_strain kelvin_voigt_strain'
  []
[]
