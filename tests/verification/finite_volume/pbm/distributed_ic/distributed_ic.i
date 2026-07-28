# Population balance, 30 bins, distributed Gaussian initial condition with a
# nearest neighbor (parent j -> child j-1) mass conserving daughter matrix.
# This case checks that the assembled pipeline reproduces the initial condition
# exactly at step 0 (a wiring / round trip check); the trajectory still has to
# integrate stably through every step.
[Tensors]
  [rho_val]
    type = Python
    expr = 'Scalar(torch.ones(30), sub_batch_ndim=1)'
  []
  [v_val]
    type = Python
    expr = 'Scalar(torch.arange(1.0, 31.0), sub_batch_ndim=1)'
  []
  [dv_val]
    type = Python
    expr = 'Scalar(torch.ones(30), sub_batch_ndim=1)'
  []
  # Sink at bin 0 (gamma_0 = 0); every other bin fragments at a constant rate.
  [gamma_val]
    type = Python
    expr = 'Scalar(torch.cat([torch.zeros(1), torch.full((29,), 0.2)]), sub_batch_ndim=1)'
  []
  # Nearest neighbor daughter matrix: parent j deposits into child j-1 on the
  # superdiagonal (k = j-1), with p[j-1, j] = v_j / v_{j-1} for mass conservation
  # (dv = rho = 1).
  [p_val]
    type = Python
    expr = 'Scalar(torch.diag(torch.arange(2.0, 31.0) / torch.arange(1.0, 30.0), 1), sub_batch_ndim=2)'
  []
  # Gaussian initial condition centered at bin 15.
  [ic]
    type = Python
    expr = 'Scalar(torch.exp(-0.5 * ((torch.arange(30.0) - 15.0) / 4.0) ** 2), sub_batch_ndim=1)'
  []
  [time]
    type = Python
    expr = 'linspace(Scalar(0.0).dynamic_batch, Scalar(1.0).dynamic_batch, 50)'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'time'
    ic_Scalar_names = 'u'
    ic_Scalar_values = 'ic'
  []
  [verification]
    type = Verification
    driver = 'driver'
    Scalar_names = 'output.u'
    Scalar_values = 'ic'
    atol = 1e-6
    rtol = 1e-6
    time_steps = '0'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'u'
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
  [rho_param]
    type = ScalarParameterToVariable
    from = 'rho_val'
    to = 'rho'
  []
  [v_param]
    type = ScalarParameterToVariable
    from = 'v_val'
    to = 'v'
  []
  [dv_param]
    type = ScalarParameterToVariable
    from = 'dv_val'
    to = 'dv'
  []
  [gamma_param]
    type = ScalarParameterToVariable
    from = 'gamma_val'
    to = 'gamma'
  []
  [p_param]
    type = ScalarParameterToVariable
    from = 'p_val'
    to = 'p'
  []
  [frag_flux]
    type = FiniteVolumeFragmentationFlux
    cell_density = 'rho'
    cell_volume = 'v'
    cell_width = 'dv'
    fragmentation_rate = 'gamma'
    breakage_matrix = 'p'
    flux_operator = 'M'
  []
  [flux]
    type = IntermediateLinearContraction
    operator = 'M'
    field = 'u'
    out = 'J'
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
    dx = 'dv_val'
    grad_u = 'u_rate'
  []
  [integrate_u]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'u'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'rho_param v_param dv_param gamma_param p_param frag_flux flux left_bc right_bc flux_divergence integrate_u'
  []
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_Scalar = 'u'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
[]
