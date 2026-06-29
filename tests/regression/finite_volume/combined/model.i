# neml2
[Tensors]
  [edges]
    # Coarsened for regression-test runtime: 201 -> 51 (50 cells). Full
    # resolution lives under tests/regression/; here we only detect drift.
    type = Python
    expr = 'linspace(Scalar(0.0).sub_batch, Scalar(1.0).sub_batch, 51)'
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
  [ic]
    type = Python
    expr = 'Scalar(1.0 * torch.exp(-0.5 * ((centers.data - 0.25) / 0.05) ** 2), sub_batch_ndim=1)'
  []
  [time]
    type = Python
    expr = 'linspace(Scalar(0.0).dynamic_batch, Scalar(1.0).dynamic_batch, 25)'
  []
  [D_cells]
    type = Python
    expr = 'Scalar(0.5 * torch.ones_like(centers.data), sub_batch_ndim=1)'
  []
  [v_cells]
    type = Python
    expr = 'Scalar(0.4 * torch.ones_like(centers.data), sub_batch_ndim=1)'
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
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Models]
  [diffusivity]
    type = LinearlyInterpolateToCellEdges
    cell_values = 'D_cells'
    cell_centers = 'centers'
    cell_edges = 'edges'
    edge_values = 'D'
  []
  [advection_velocity]
    type = LinearlyInterpolateToCellEdges
    cell_values = 'v_cells'
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
    weights = '-0.05'
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
