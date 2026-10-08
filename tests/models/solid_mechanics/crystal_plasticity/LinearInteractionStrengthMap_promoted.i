# Model unit test: LinearInteractionStrengthMap with a PROMOTED interaction matrix
# (so ModelUnitTest's JVP covers d(tau)/d(h) as well as d(tau)/d(slip_hardening)).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'slip_hardening h_in'
    input_Scalar_values = 'phi hmat'
    output_Scalar_names = 'slip_strengths'
    output_Scalar_values = 'tau'
    derivative_rel_tol = 0
    derivative_abs_tol = 5e-6
  []
[]

[Tensors]
  [phi]
    type = Python
    expr = 'Scalar([1.0, 4.0, 9.0, 16.0]).sub_batch.retag(1)'
  []
  [hmat]
    type = Python
    expr = 'Scalar(torch.tensor([[1.0,1.0,0.5,0.5],[1.0,1.0,0.5,0.5],[0.5,0.5,1.0,1.0],[0.5,0.5,1.0,1.0]], dtype=torch.float64)).sub_batch.retag(2)'
  []
  [tau]
    type = Python
    expr = 'Scalar(5.0 + torch.tensor([[1.0,1.0,0.5,0.5],[1.0,1.0,0.5,0.5],[0.5,0.5,1.0,1.0],[0.5,0.5,1.0,1.0]], dtype=torch.float64) @ torch.tensor([1.0,4.0,9.0,16.0], dtype=torch.float64)).sub_batch.retag(1)'
  []
[]

[Models]
  [model]
    type = LinearInteractionStrengthMap
    constant_strength = 5.0
    interaction_matrix = 'h_in'
  []
[]
