# Model unit test: LinearInteractionHardeningRule with a PROMOTED interaction matrix
# (so ModelUnitTest's JVP covers d(rate)/d(M) as well as d(rate)/d(gamma_dot)).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'slip_hardening slip_rates m_in'
    input_Scalar_values = 'h0 gdot mmat'
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
    type = Python
    expr = 'Scalar(torch.tensor([[1.0,1.0,0.5,0.5],[1.0,1.0,0.5,0.5],[0.5,0.5,1.0,1.0],[0.5,0.5,1.0,1.0]], dtype=torch.float64)).sub_batch.retag(2)'
  []
  [rate]
    type = Python
    expr = 'Scalar(torch.tensor([[1.0,1.0,0.5,0.5],[1.0,1.0,0.5,0.5],[0.5,0.5,1.0,1.0],[0.5,0.5,1.0,1.0]], dtype=torch.float64) @ torch.tensor([1.0,4.0,9.0,16.0], dtype=torch.float64)).sub_batch.retag(1)'
  []
[]

[Models]
  [model]
    type = LinearInteractionHardeningRule
    interaction_matrix = 'm_in'
  []
[]
