# neml2
# Stress relaxation of a Zener (Standard Linear Solid) viscoelastic element under a ramped strain
# loading. With a single Maxwell branch in parallel with an equilibrium spring, the stress
# overshoots during loading and relaxes to the equilibrium value E_inf * strain when the strain is
# held fixed.
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
  [zener]
    type = ZenerElement
    equilibrium_modulus = 1000
    maxwell_modulus = 5000
    maxwell_viscosity = 100
  []
  [integrate_Ev]
    type = SR2BackwardEulerTimeIntegration
    variable = 'viscous_strain'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'zener integrate_Ev'
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
    models = 'update zener'
    additional_outputs = 'viscous_strain'
  []
[]
