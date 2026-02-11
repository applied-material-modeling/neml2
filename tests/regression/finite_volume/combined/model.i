[Tensors]
  [edges]
    type = LinspaceScalar
    start = 0.0
    end = 1.0
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
    width = 0.05
    height = 1.0
    center = 0.25
  []
  [time]
    type = LinspaceScalar
    start = 0.0
    end = 1.0
    nstep = 25
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
    cell_values = 0.5
    cell_centers = 'centers'
    cell_edges = 'edges'
    edge_values = 'state/D'
  []
  [advection_velocity]
    type = LinearlyInterpolateToCellEdges
    cell_values = 0.4
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
    coefficients = '-0.05'
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