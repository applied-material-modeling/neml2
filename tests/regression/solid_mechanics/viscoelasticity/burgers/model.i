# neml2
# Creep-like response of a Burgers viscoelastic element under a ramped strain history. The Burgers
# model has two internal strains (Maxwell dashpot viscous strain and Kelvin-Voigt branch strain)
# that are solved for via an implicit Newton update at each time step.
[Tensors]
  [end_time]
    type = Python
    expr = 'Scalar(torch.logspace(-1, 3, 10, dtype=torch.float64))'
  []
  [times]
    type = Python
    expr = 'Scalar(torch.stack([torch.linspace(0, t.item(), 100, dtype=torch.float64) for t in end_time.data], dim=-1))'
  []
  [strains]
    # Mandel-fill (10, 6) max_strain = (exx, eyy, ezz, 0, 0, 0) = (0.01, -0.005, -0.005, 0, 0, 0),
    # then linspace ramp from 0 to max across 100 timesteps -> shape (100, 10, 6).
    type = Python
    expr = 'SR2(torch.tensor([0.01, -0.005, -0.005, 0.0, 0.0, 0.0], dtype=torch.float64).reshape(1, 1, 6).expand(100, 10, 6) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).reshape(100, 1, 1))'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_SR2_names = 'strain'
    prescribed_SR2_values = 'strains'
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
