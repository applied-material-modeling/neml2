# Population balance regression: grinding.
# 20 bins, uniform density; larger particles fragment faster (gamma ~ bin index,
# with a sink at bin 0). The daughter matrix splits a parent's mass equally over
# all smaller bins (p[k,j] = v_j / (j v_k) for k < j), which is mass conserving
# for uniform density. A coarse Gaussian feed grinds toward the fines over time.
[Tensors]
  [cell_volume_val]
    type = Python
    expr = 'Scalar(torch.arange(1.0, 21.0), sub_batch_ndim=1)'
  []
  [cell_density_val]
    type = Python
    expr = 'Scalar(torch.ones(20), sub_batch_ndim=1)'
  []
  [cell_width_val]
    type = Python
    expr = 'Scalar(torch.ones(20), sub_batch_ndim=1)'
  []
  # Fragmentation rate increases with size; bin 0 is a sink (gamma_0 = 0).
  [fragmentation_rate_val]
    type = Python
    expr = 'Scalar(0.05 * torch.arange(20.0), sub_batch_ndim=1)'
  []
  # Equal-mass split over smaller bins: p[k,j] = v_j / (j v_k) for k < j (strict
  # upper triangle), zero elsewhere. The column index j is the number of smaller
  # bins; clamp(min=1) only guards the masked-out j=0 column.
  [breakage_matrix_val]
    type = Python
    expr = 'Scalar(torch.triu(torch.ones(20, 20), 1) * ((cell_volume_val.data / torch.arange(20.0).clamp(min=1.0)).unsqueeze(0) / cell_volume_val.data.unsqueeze(1)), sub_batch_ndim=2)'
  []
  # Coarse Gaussian feed centered at bin 14.
  [ic]
    type = Python
    expr = 'Scalar(torch.exp(-0.5 * ((torch.arange(20.0) - 14.0) / 3.0) ** 2), sub_batch_ndim=1)'
  []
  [time]
    type = Python
    expr = 'linspace(Scalar(0.0).dynamic_batch, Scalar(5.0).dynamic_batch, 25)'
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
