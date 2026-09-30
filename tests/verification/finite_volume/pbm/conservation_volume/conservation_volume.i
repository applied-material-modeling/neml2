# Population balance, 10 bins, non-uniform density -- strongest check: conserves
# both total mass (sum u_i dv_i) and total volume (sum u_i/rho_i dv_i). Mass is
# conserved by the flux form for any p; volume needs a p designed for it. Density
# is non-monotone (light sink bin 0, heavy sink bin 1) so a non-trivial
# volume conserving p exists. Bins 0,1 are stable sinks (gamma=0); coarser bins
# fragment only into them with p[0,j]=0.6 v_j, p[1,j]=0.2 v_j, the unique
# solution (dv=1) of
#   mass:   sum_k rho_k v_k p[k,j] = rho_j v_j
#   volume: sum_k v_k p[k,j] (1 - rho_k/rho_j) = 0
# The [Verification] block checks only the step-0 round trip; the invariants are
# asserted across the trajectory by the companion test_conservation_volume.py.
[Tensors]
  [rho_val]
    type = Python
    expr = 'Scalar(torch.cat([torch.tensor([0.5, 3.0]), torch.full((8,), 1.5)]), sub_batch_ndim=1)'
  []
  [v_val]
    type = Python
    expr = 'Scalar(torch.arange(1.0, 11.0), sub_batch_ndim=1)'
  []
  [dv_val]
    type = Python
    expr = 'Scalar(torch.ones(10), sub_batch_ndim=1)'
  []
  [gamma_val]
    type = Python
    expr = 'Scalar(torch.cat([torch.zeros(2), torch.full((8,), 0.5)]), sub_batch_ndim=1)'
  []
  # Rows 0 and 1 hold the daughter fractions into the two sink bins; the
  # remaining rows are zero. p[0,j]=0.6 v_j, p[1,j]=0.2 v_j for j>=2.
  [p_val]
    type = Python
    expr = 'Scalar(torch.cat([torch.stack([torch.cat([torch.zeros(2), 0.6 * torch.arange(3.0, 11.0)]), torch.cat([torch.zeros(2), 0.2 * torch.arange(3.0, 11.0)])]), torch.zeros(8, 10)]), sub_batch_ndim=2)'
  []
  [ic]
    type = Python
    expr = 'Scalar(torch.cat([torch.zeros(6), torch.ones(4)]), sub_batch_ndim=1)'
  []
  [time]
    type = Python
    expr = 'linspace(Scalar(0.0).dynamic_batch, Scalar(2.0).dynamic_batch, 100)'
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
