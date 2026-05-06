D = 0.0
v = 0.5
l = -0.0

c0 = 1.0
w = 0.05
center = 0.25

t = 1.0


[Tensors]
  [edges]
    type = LinspaceScalar
    start = 0.0
    end = 1.0
    nstep = 101
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