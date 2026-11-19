# Model unit test: LinearInteractionHardeningRule
# slip_hardening_rate_i = sum_j M_ij |gamma_dot_j|  (interaction in the hardening rate)
# slip_hardening is an input only so the integrator wires the state; the rate does not
# depend on it (structural-zero derivative). gamma_dot carries mixed signs to exercise
# the sign() in d(rate)/d(gamma_dot) = M_ij sign(gamma_dot_j).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'slip_hardening slip_rates'
    input_Scalar_values = 'h0 gdot'
    output_Scalar_names = 'slip_hardening_rate'
    output_Scalar_values = 'rate'
    derivative_rel_tol = 0
    derivative_abs_tol = 5e-6
  []
[]

[Tensors]
  [h0]
    type = Python
    expr = 'Scalar([0.0, 0.0, 0.0, 0.0]).sub_batch.retag(1)'
  []
  [gdot]
    type = Python
    expr = 'Scalar([1.0, -4.0, 9.0, -16.0]).sub_batch.retag(1)'
  []
  [mmat]
    type = SquareMatrix
    fill = block
    blocks = '2 2'
    data = '1.0 0.5 0.5 1.0'
  []
  [rate]
    type = Python
    expr = 'Scalar(torch.tensor([[1.0,1.0,0.5,0.5],[1.0,1.0,0.5,0.5],[0.5,0.5,1.0,1.0],[0.5,0.5,1.0,1.0]], dtype=torch.float64) @ torch.tensor([1.0,4.0,9.0,16.0], dtype=torch.float64)).sub_batch.retag(1)'
  []
[]

[Models]
  [model]
    type = LinearInteractionHardeningRule
    interaction_matrix = 'mmat'
  []
[]
