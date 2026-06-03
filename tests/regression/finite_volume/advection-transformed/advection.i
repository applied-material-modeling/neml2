# neml2
# 1D linear advection of a Gaussian pulse on a semi-infinite scaled grid.
# 100 cells (101 edges) with a coordinate transform x_hat = s x / (1 - x) so
# the [0, 1) parameter axis maps to [0, infty) physical space. Concentration
# is integrated by backward Euler with upwinded fluxes and zero-flux Dirichlet
# BCs at both ends, then unscaled at the end of each step.
D = 0.0
v = 0.5
l = -0.0

c0 = 1.0
w = 0.05
center = 0.25

t = 1.0

[Tensors]
  [edges]
    type = Python
    expr = 'Scalar(torch.linspace(0.0, 1.0, 101, dtype=torch.float64), sub_batch_ndim=1)'
  []
  [centers]
    type = Python
    expr = 'Scalar(0.5 * (edges.data[..., :-1] + edges.data[..., 1:]), sub_batch_ndim=1)'
  []
  [dx_centers]
    type = Python
    expr = 'Scalar(centers.data[..., 1:] - centers.data[..., :-1], sub_batch_ndim=1)'
  []
  [dx]
    type = Python
    expr = 'Scalar(edges.data[..., 1:] - edges.data[..., :-1], sub_batch_ndim=1)'
  []

  [scale_factor]
    type = Python
    expr = 'Scalar(torch.tensor(0.1, dtype=torch.float64))'
  []

  [true_centers]
    type = Python
    expr = 'Scalar(scale_factor.data * centers.data / (1.0 - centers.data), sub_batch_ndim=1)'
  []

  [center_jacobian]
    type = Python
    expr = 'Scalar(scale_factor.data / (1.0 - centers.data) ** 2, sub_batch_ndim=1)'
  []
  [center_inverse_jacobian]
    type = Python
    expr = 'Scalar((1.0 - centers.data) ** 2 / scale_factor.data, sub_batch_ndim=1)'
  []

  [cell_velocity]
    type = Python
    expr = 'Scalar(torch.tensor(0.5, dtype=torch.float64))'
  []

  [unscaled_ic]
    type = Python
    expr = 'Scalar(1.0 * torch.exp(-0.5 * ((true_centers.data - 0.25) / 0.05) ** 2), sub_batch_ndim=1)'
  []

  [ic]
    type = Python
    expr = 'Scalar(unscaled_ic.data * center_jacobian.data, sub_batch_ndim=1)'
  []

  [time]
    type = Python
    expr = 'Scalar(torch.linspace(0.0, 1.0, 500, dtype=torch.float64))'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'time'
    ic_Scalar_names = 'concentration'
    ic_Scalar_values = 'ic'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
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
  [velocity]
    type = ScalarConstantParameter
    value = 'cell_velocity'
  []
  [scaled_cell_velocity]
    type = ScalarMultiplication
    from = 'velocity'
    scaling = 'center_inverse_jacobian'
    to = 'scaled_cell_velocity'
  []
  [advection_velocity]
    type = LinearlyInterpolateToCellEdges
    cell_values = 'scaled_cell_velocity'
    cell_centers = 'centers'
    cell_edges = 'edges'
    edge_values = 'v_edge'
  []
  [advective_flux]
    type = FiniteVolumeUpwindedAdvectiveFlux
    u = 'concentration'
    v_edge = 'v_edge'
    flux = 'J'
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
    grad_u = 'concentration_rate'
  []
  [integrate_u]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'concentration'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'velocity scaled_cell_velocity advection_velocity advective_flux left_bc right_bc flux_divergence integrate_u'
  []
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_Scalar = 'concentration'
  []
  [model_scaled]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [unscale]
    type = ScalarMultiplication
    from = 'concentration'
    scaling = 'center_inverse_jacobian'
    to = 'true_concentration'
  []
  [model]
    type = ComposedModel
    models = 'model_scaled unscale'
    additional_outputs = 'concentration'
  []
[]
