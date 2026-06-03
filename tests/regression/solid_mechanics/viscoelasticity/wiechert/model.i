# neml2
# Stress relaxation of a Wiechert (generalized Maxwell) element with two branches under a ramped
# strain. Each Maxwell branch contributes its own relaxation time scale, and the equilibrium spring
# carries the long-time stress.
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
    type = WiechertElement
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
