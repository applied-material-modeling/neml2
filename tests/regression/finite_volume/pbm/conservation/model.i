# Population balance regression: mass and volume conserving fragmentation with
# non-uniform density. 10 bins; density is non-monotone (light sink bin 0, heavy
# sink bin 1) so a non-trivial volume conserving daughter matrix exists. Coarser
# bins fragment only into the two stable sinks (gamma_0 = gamma_1 = 0) with
# p[0,j] = 0.6 v_j, p[1,j] = 0.2 v_j -- the solution of the discrete mass and
# volume constraints. Companion of the conservation_volume verification case;
# here we just pin the trajectory against drift.
[Tensors]
  [cell_density_val]
    type = Python
    expr = 'Scalar(torch.cat([torch.tensor([0.5, 3.0]), torch.full((8,), 1.5)]), sub_batch_ndim=1)'
  []
  [cell_volume_val]
    type = Python
    expr = 'Scalar(torch.arange(1.0, 11.0), sub_batch_ndim=1)'
  []
  [cell_width_val]
    type = Python
    expr = 'Scalar(torch.ones(10), sub_batch_ndim=1)'
  []
  [fragmentation_rate_val]
    type = Python
    expr = 'Scalar(torch.cat([torch.zeros(2), torch.full((8,), 0.5)]), sub_batch_ndim=1)'
  []
  [breakage_matrix_val]
    type = Python
    expr = 'Scalar(torch.cat([torch.stack([torch.cat([torch.zeros(2), 0.6 * torch.arange(3.0, 11.0)]), torch.cat([torch.zeros(2), 0.2 * torch.arange(3.0, 11.0)])]), torch.zeros(8, 10)]), sub_batch_ndim=2)'
  []
  [ic]
    type = Python
    expr = 'Scalar(torch.cat([torch.zeros(6), torch.ones(4)]), sub_batch_ndim=1)'
  []
  [time]
    type = Python
    expr = 'linspace(Scalar(0.0).dynamic_batch, Scalar(2.0).dynamic_batch, 25)'
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
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
    rtol = 1e-5
    atol = 1e-8
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
  [cell_density]
    type = ScalarParameterToVariable
    from = 'cell_density_val'
    to = 'cell_density'
  []
  [cell_volume]
    type = ScalarParameterToVariable
    from = 'cell_volume_val'
    to = 'cell_volume'
  []
  [cell_width]
    type = ScalarParameterToVariable
    from = 'cell_width_val'
    to = 'cell_width'
  []
  [fragmentation_rate]
    type = ScalarParameterToVariable
    from = 'fragmentation_rate_val'
    to = 'fragmentation_rate'
  []
  [breakage_matrix]
    type = ScalarParameterToVariable
    from = 'breakage_matrix_val'
    to = 'breakage_matrix'
  []
  [frag_flux]
    type = FiniteVolumeFragmentationFlux
    cell_density = 'cell_density'
    cell_volume = 'cell_volume'
    cell_width = 'cell_width'
    fragmentation_rate = 'fragmentation_rate'
    breakage_matrix = 'breakage_matrix'
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
    dx = 'cell_width_val'
    grad_u = 'u_rate'
  []
  [integrate_u]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'u'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'cell_density cell_volume cell_width fragmentation_rate breakage_matrix frag_flux flux left_bc right_bc flux_divergence integrate_u'
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
