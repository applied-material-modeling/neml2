# neml2
# Native port of tests/verification/solid_mechanics/viscoelasticity/maxwell/maxwell.i.
# Verification of LinearDashpot against the closed-form solution for the dashpot under
# constant prescribed stress. dε_v/dt = σ/η so ε_v(t) = (σ/η) t for σ held from t = 0.
# All reference tensors are analytical, constructed inline via [Tensors] Python expressions.
#
# Mandel scaling note: the C++ FillSR2 applies sqrt(2) to slots 3..5 (shear) internally.
# The native SR2 wrapper does not; we apply sqrt(2) explicitly below to keep storage equivalent.

[Tensors]
  # LinspaceScalar(0, 1.0, 21) -> shape (21,)
  [times]
    type = Python
    expr = 'Scalar.linspace(0.0, 1.0, 21)'
  []

  # Constant stress σ = (100, -50, -50, 20, -10, 5) (physical); Mandel scaling applied
  # on shear slots so storage matches C++ FillSR2 output.
  [stress_value]
    type = Python
    expr = 'SR2(torch.tensor([100.0, -50.0, -50.0, 20.0 * (2.0 ** 0.5), -10.0 * (2.0 ** 0.5), 5.0 * (2.0 ** 0.5)], dtype=torch.float64))'
  []

  # stress_history = LinspaceSR2(stress_value, stress_value, 21) -> shape (21, 6).
  # Linear interpolation from x to x is just x repeated; build via broadcast over a
  # length-21 axis.
  [stress_history]
    type = Python
    expr = 'SR2(stress_value.data.unsqueeze(0) + (stress_value.data - stress_value.data).unsqueeze(0) * torch.linspace(0.0, 1.0, 21, dtype=torch.float64).reshape(21, 1))'
  []

  # Ev_final = stress_value / 100 (the dashpot viscosity), physical values
  # (1.0, -0.5, -0.5, 0.2, -0.1, 0.05); Mandel scaling on shear.
  [Ev_final]
    type = Python
    expr = 'SR2(torch.tensor([1.0, -0.5, -0.5, 0.2 * (2.0 ** 0.5), -0.1 * (2.0 ** 0.5), 0.05 * (2.0 ** 0.5)], dtype=torch.float64))'
  []

  # Ev_reference = LinspaceSR2(0, Ev_final, 21) -> shape (21, 6).
  [Ev_reference]
    type = Python
    expr = 'SR2(Ev_final.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 21, dtype=torch.float64).reshape(21, 1))'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'stress'
    force_SR2_values = 'stress_history'
    save_as = 'result.pt'
  []
  [verification]
    type = Verification
    driver = 'driver'
    SR2_names = 'output.viscous_strain'
    SR2_values = 'Ev_reference'
    rtol = 1e-6
    atol = 1e-8
  []
[]

[Models]
  [maxwell]
    type = LinearDashpot
    viscosity = 100
  []
  [integrate_Ev]
    type = SR2BackwardEulerTimeIntegration
    variable = 'viscous_strain'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'maxwell integrate_Ev'
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
    models = 'update'
    additional_outputs = 'viscous_strain'
  []
[]
