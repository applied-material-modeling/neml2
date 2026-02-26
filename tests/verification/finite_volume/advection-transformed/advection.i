D = 0.0
v = 0.5
l = -0.0

c0 = 1.0
w = 0.05
center = 0.25

t = 1.0

# D_eff = D + |v| * \delta x / 2

# Final height = c0 * w / sqrt(2 * D_eff * t + w^2) * exp(-l*t)
h_final = 0.666667

# Final width = sqrt(2 * D_eff * t + w^2)
final_width = 0.0750

# Final center = center + v * t
final_center = 0.75

[Tensors]
  [edges]
    type = LinspaceScalar
    start = 0.0
    end = 1.0
    nstep = 801
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

  [scale_factor]
    type = Scalar
    values = 0.1
  []

  [true_centers]
    type = SemiInfiniteScalingScalar
    x = 'centers'
    s = 'scale_factor'
  []

  [center_jacobian]
    type = SemiInfiniteScalingJacobianScalar
    x = 'centers'
    s = 'scale_factor'
  []
  [center_inverse_jacobian]
    type = InverseSemiInfiniteScalingJacobianScalar
    x = 'centers'
    s = 'scale_factor'
  []

  [cell_velocity]
    type = Scalar
    values = ${v}
  []

  [unscaled_ic]
    type = GaussianScalar 
    points = 'true_centers'
    width = ${w}
    height = ${c0}
    center = ${center}
  []

  [ic]
    type = ProductUserTensorScalar
    a = 'unscaled_ic'
    b = 'center_jacobian'
  []

  [time]
    type = LinspaceScalar
    start = 0.0
    end = ${t}
    nstep = 500
  []

  [result]
    type = GaussianScalar 
    points = 'true_centers'
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
#  [verification]
#    type = VTestVerification
#    driver = 'driver'
#    Scalar_names = 'output.state/true_concentration'
#    Scalar_values = 'result'
#    atol = 1e-2
#    rtol = 1e-1 # The time integration also adds diffusion...
#    time_steps = '499'
#  []
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
  [velocity]
    type = ScalarConstantParameter
    value = 'cell_velocity'
  []
  [scaling]
    type = ScalarConstantParameter
    value = 'center_inverse_jacobian'
  []
  [scaled_cell_velocity]
    type = ScalarMultiplication
    from_var = 'parameters/velocity parameters/scaling'
    to_var = 'state/internal/scaled_cell_velocity'
  []
  [advection_velocity]
    type = LinearlyInterpolateToCellEdges
    cell_values = 'state/internal/scaled_cell_velocity'
    cell_centers = 'centers'
    cell_edges = 'edges'
    edge_values = 'state/v_edge'
  []
  [advective_flux]
    type = FiniteVolumeUpwindedAdvectiveFlux
    u = 'state/concentration'
    v_edge = 'state/v_edge'
    flux = 'state/J'
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
    grad_u = 'state/concentration_rate'
  []
  [integrate_u]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/concentration'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'velocity scaling scaled_cell_velocity advection_velocity advective_flux left_bc right_bc flux_divergence integrate_u'
  []
  [model_scaled]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
  [unscale]
    type = ScalarMultiplication
    from_var = 'state/concentration parameters/scaling'
    to_var = 'state/true_concentration'
  []
  [model]
    type = ComposedModel
    models = 'model_scaled unscale scaling'
    additional_outputs = 'state/concentration'
   []
[]