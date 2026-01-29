[Tensors]
  [dx]
    type = FullScalar
    batch_shape = '(100,)'
    value = 0.01
    intermediate_dimension = 1
  []
  [ic]
    type = GaussianScalar 
    start = 0.005
    end = 0.995
    width = 0.1
    height = 1.0
    nstep = 100
    group = 'intermediate'
    dim = 0
    center = 0.25
  []
  [time]
    type = LinspaceScalar
    start = 0.0
    end = 1.0
    nstep = 20
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
[]


[Solvers]
  [newton]
    type = Newton
    linear_solver = 'lu'
    verbose = true
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [diffusive_flux]
      type = DiffusiveFlux
      u = 'state/concentration'
      D = 1.0
  []
  [advective_flux]
      type = AdvectiveFlux
      u = 'state/concentration'
      v = 0.5
  []
  [reaction]
      type = LinearReaction
      lambda = 0.1
      u = 'state/concentration'
  []
  [total_flux]
    type = ScalarLinearCombination
    from_var = 'state/J_diffusion state/J_advection'
    to_var = 'state/J'
    coefficients = '1 1'
  []
  [left_bc]
    type = TransportBoundaryCondition
    input = 'state/J'
    bc_value = 0.0
    side = 'left'
  []
  [right_bc]
    type = TransportBoundaryCondition
    input = 'state/J_with_bc_left'
    bc_value = 0.0
    side = 'right'
  []
  [rate_of_change]
    type = CellRateOfChange
    flux = 'state/J_with_bc_left_with_bc_right '
    cell_size = 'dx'
    reaction = 'state/R'
    rate = 'state/concentration_rate'
  []
  [integrate_u]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/concentration'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'diffusive_flux advective_flux reaction total_flux left_bc right_bc rate_of_change integrate_u'
  []
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
[]