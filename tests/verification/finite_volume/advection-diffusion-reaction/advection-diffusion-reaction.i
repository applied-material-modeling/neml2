D = 0.01
v = 0.5
l = -0.1

c0 = 1.0
w = 0.05
center = 0.25

t = 1.0

# D_eff = D + |v| * \delta x / 2

# Final height = c0 * w / sqrt(2 * D_eff * t + w^2) * exp(-l*t)
h_final = 0.2826237

# Final width = sqrt(2 * D_eff * t + w^2)
final_width = 0.1600781

# Final center = center + v * t
final_center = 0.75

[Tensors]
  [edges]
    type = LinspaceScalar
    start = 0.0
    end = 1.25
    nstep = 201
    dim = 0
    group = 'intermediate'
  []
  [centers]
    type = CenterScalar
    points = 'edges'
  []
  [dx_centers]
    type = DifferenceScalar
    points = 'centers'
  []
  [dx]
    type = DifferenceScalar
    points = 'edges'
  []
  [ic]
    type = GaussianScalar 
    points = 'centers'
    width = ${w}
    height = ${c0}
    center = ${center}
  []
  [time]
    type = LinspaceScalar
    start = 0.0
    end = ${t}
    nstep = 500
  []

  [result]
    type = GaussianScalar 
    points = 'centers'
    width = ${final_width} 
    height = ${h_final}
    center = ${final_center}
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'time'
    ic_Scalar_names = 'state/concentration'
    ic_Scalar_values = 'ic'
    save_as = 'result.pt'
  []
  [verification]
    type = VTestVerification
    driver = 'driver'
    Scalar_names = 'output.state/concentration'
    Scalar_values = 'result'
    atol = 1e-2
    rtol = 1e-2 # The time integration also adds diffusion...
    time_steps = '499'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
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
  [diffusivity]
    type = LinearlyInterpolateToCellEdges
    cell_values = ${D}
    cell_centers = 'centers'
    cell_edges = 'edges'
    edge_values = 'state/D'
  []
  [advection_velocity]
    type = LinearlyInterpolateToCellEdges
    cell_values = ${v}
    cell_centers = 'centers'
    cell_edges = 'edges'
    edge_values = 'state/v_edge'
  []
  [diffusive_flux]
      type = FiniteVolumeGradient
      u = 'state/concentration'
      prefactor = 'state/D'
      dx = 'dx_centers'
  []
  [advective_flux]
      type = FiniteVolumeUpwindedAdvectiveFlux
      u = 'state/concentration'
      v_edge = 'state/v_edge'
  []
  [reaction]
      type = ScalarLinearCombination
      from_var = 'state/concentration'
      to_var = 'state/R'
      coefficients = '${l}'
  []
  [total_flux]
    type = ScalarLinearCombination
    from_var = 'state/grad_u state/J_advection'
    to_var = 'state/J'
    coefficients = '1 1'
  []
  [left_bc]
    type = FiniteVolumeAppendBoundaryCondition
    input = 'state/J'
    bc_value = 0.0
    side = 'left'
  []
  [right_bc]
    type = FiniteVolumeAppendBoundaryCondition
    input = 'state/J_with_bc_left'
    bc_value = 0.0
    side = 'right'
  []
  [flux_divergence]
    type = FiniteVolumeGradient
    u = 'state/J_with_bc_left_with_bc_right'
    dx = 'dx'
    grad_u = 'state/flux_div'
  []
  [rate_of_change]
    type = ScalarLinearCombination
    from_var = 'state/R state/flux_div'
    to_var = 'state/concentration_rate'
    coefficients = '1 1'
  []
  [integrate_u]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/concentration'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'diffusivity advection_velocity diffusive_flux advective_flux reaction total_flux left_bc right_bc flux_divergence rate_of_change integrate_u'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
[]