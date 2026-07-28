# Population balance, two bins, closed-form check.
# Uniform density; bin 0 is a sink (gamma_0 = 0); bin 1 fragments into bin 0 at
# rate gamma_1. The mass conserving daughter matrix reduces the discrete system
# to du_1/dt = -gamma_1 u_1, so u_1(t) = u_1(0) exp(-gamma_1 t) and the mass lost
# by bin 1 accumulates in bin 0. Reference is the closed form at the final step.
g1 = 1.0
t = 1.0

[Tensors]
  [rho_val]
    type = Python
    expr = 'Scalar(torch.tensor([1.0, 1.0]), sub_batch_ndim=1)'
  []
  [v_val]
    type = Python
    expr = 'Scalar(torch.tensor([1.0, 2.0]), sub_batch_ndim=1)'
  []
  [dv_val]
    type = Python
    expr = 'Scalar(torch.tensor([1.0, 1.0]), sub_batch_ndim=1)'
  []
  [gamma_val]
    type = Python
    expr = 'Scalar(torch.tensor([0.0, ${g1}]), sub_batch_ndim=1)'
  []
  # Daughter matrix p[k,j]: parent bin 1 breaks entirely into bin 0. The value
  # p[0,1]=2 satisfies discrete mass conservation dv_0 rho_0 v_0 p_01 = rho_1 v_1.
  [p_val]
    type = Python
    expr = 'Scalar(torch.tensor([[1.0, 2.0], [0.0, 0.0]]), sub_batch_ndim=2)'
  []
  [ic]
    type = Python
    expr = 'Scalar(torch.tensor([0.0, 1.0]), sub_batch_ndim=1)'
  []
  [time]
    type = Python
    expr = 'linspace(Scalar(0.0).dynamic_batch, Scalar(${t}).dynamic_batch, 200)'
  []
  # u_1(t) = exp(-g1 t) = exp(-1) = 0.36787944; u_0(t) = 1 - u_1(t) by mass
  # conservation = 0.63212056.
  [result]
    type = Python
    expr = 'Scalar(torch.tensor([0.63212056, 0.36787944]), sub_batch_ndim=1)'
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
    Scalar_values = 'result'
    atol = 1e-2
    rtol = 1e-2
    time_steps = '199'
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
