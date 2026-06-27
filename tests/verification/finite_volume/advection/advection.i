# neml2
# Native port of tests/verification/finite_volume/advection/advection.i.
# Pure advection of a Gaussian pulse on a uniform cell-edge grid; the final
# concentration profile is compared against the analytical advected Gaussian
# (broadened by the numerical-diffusion contribution from upwinding).
D = 0.0
v = 0.5
l = -0.0

c0 = 1.0
w = 0.05
center = 0.25

t = 1.0

# Final height = c0 * w / sqrt(2 * D_eff * t + w^2) * exp(-l*t)
h_final = 0.666667

# Final width = sqrt(2 * D_eff * t + w^2)
final_width = 0.0750

# Final center = center + v * t
final_center = 0.75

[Tensors]
  [edges]
    type = Python
    expr = 'Scalar.linspace(0.0, 1.25, 201).sub_batch.retag(1)'
  []
  [centers]
    type = Python
    expr = 'Scalar(0.5 * (edges.data[..., 1:] + edges.data[..., :-1]), sub_batch_ndim=1)'
  []
  [dx_centers]
    type = Python
    expr = 'Scalar(centers.data[..., 1:] - centers.data[..., :-1], sub_batch_ndim=1)'
  []
  [dx]
    type = Python
    expr = 'Scalar(edges.data[..., 1:] - edges.data[..., :-1], sub_batch_ndim=1)'
  []
  # The native LinearlyInterpolateToCellEdges expects a size-N Scalar for
  # cell_values; the C++ version broadcasts a 0-dim Scalar to size N
  # implicitly. Build the broadcast Scalar explicitly here.
  [D_const]
    type = Python
    expr = 'Scalar(torch.full_like(centers.data, ${D}), sub_batch_ndim=1)'
  []
  [v_const]
    type = Python
    expr = 'Scalar(torch.full_like(centers.data, ${v}), sub_batch_ndim=1)'
  []
  [ic]
    type = Python
    expr = 'Scalar(${c0} * torch.exp(-0.5 * ((centers.data - ${center}) / ${w}) ** 2), sub_batch_ndim=1)'
  []
  [time]
    type = Python
    expr = 'Scalar.linspace(0.0, ${t}, 500)'
  []

  [result]
    type = Python
    expr = 'Scalar(${h_final} * torch.exp(-0.5 * ((centers.data - ${final_center}) / ${final_width}) ** 2), sub_batch_ndim=1)'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'time'
    ic_Scalar_names = 'concentration'
    ic_Scalar_values = 'ic'
  []
  [verification]
    type = Verification
    driver = 'driver'
    Scalar_names = 'output.concentration'
    Scalar_values = 'result'
    atol = 1e-2
    rtol = 1e-1
    # The time integration also adds diffusion...
    time_steps = '499'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'concentration'
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
    cell_values = 'D_const'
    cell_centers = 'centers'
    cell_edges = 'edges'
    edge_values = 'D'
  []
  [advection_velocity]
    type = LinearlyInterpolateToCellEdges
    cell_values = 'v_const'
    cell_centers = 'centers'
    cell_edges = 'edges'
    edge_values = 'v_edge'
  []
  [diffusive_flux]
    type = FiniteVolumeGradient
    u = 'concentration'
    prefactor = 'D'
    dx = 'dx_centers'
  []
  [advective_flux]
    type = FiniteVolumeUpwindedAdvectiveFlux
    u = 'concentration'
    v_edge = 'v_edge'
  []
  [reaction]
    type = ScalarLinearCombination
    from = 'concentration'
    to = 'R'
    weights = '${l}'
  []
  [total_flux]
    type = ScalarLinearCombination
    from = 'grad_u flux'
    to = 'J'
    weights = '1 1'
  []
  [left_bc]
    type = FiniteVolumeAppendBoundaryCondition
    input = 'J'
    bc_value = 0.0
    side = 'left'
  []
  [right_bc]
    type = FiniteVolumeAppendBoundaryCondition
    input = 'J_with_bc_left'
    bc_value = 0.0
    side = 'right'
  []
  [flux_divergence]
    type = FiniteVolumeGradient
    u = 'J_with_bc_left_with_bc_right'
    dx = 'dx'
    grad_u = 'flux_div'
  []
  [rate_of_change]
    type = ScalarLinearCombination
    from = 'R flux_div'
    to = 'concentration_rate'
    weights = '1 1'
  []
  [integrate_u]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'concentration'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'diffusivity advection_velocity diffusive_flux advective_flux reaction total_flux left_bc right_bc flux_divergence rate_of_change integrate_u'
  []
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_Scalar = 'concentration'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
[]
