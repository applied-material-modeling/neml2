# Verification of LinearDashpot against the closed-form solution for the dashpot
# under constant prescribed stress.
#
# The Maxwell dashpot equation is dε_v/dt = σ/η. Under a constant stress σ held from t = 0
# the viscous strain therefore grows linearly: ε_v(t) = (σ/η) t. That linear-in-time profile
# is the analytical reference, constructed inline in [Tensors] via LinspaceSR2.

[Tensors]
  [times]
    type = LinspaceScalar
    start = 0
    end = 1.0
    nstep = 21
  []

  # Constant stress, repeated at every prescribed time step (start = end).
  [stress_value]
    type = FillSR2
    values = '100 -50 -50 20 -10 5'
  []
  [stress_history]
    type = LinspaceSR2
    start = stress_value
    end = stress_value
    nstep = 21
  []

  # Analytical viscous strain at t_final = 1.0: (σ/η) t_final = stress_value / 100.
  [Ev_final]
    type = FillSR2
    values = '1.0 -0.5 -0.5 0.2 -0.1 0.05'
  []
  [Ev_reference]
    type = LinspaceSR2
    start = 0
    end = Ev_final
    nstep = 21
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
    type = VTestVerification
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
