# Model unit test: DislocationInteractionStrengthMap
# tau_i = constant_strength + alpha*mu*b*sqrt(sum_r h_ir rho_r)
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'dislocation_density'
    input_Scalar_values = 'rho'
    output_Scalar_names = 'slip_strengths'
    output_Scalar_values = 'tau'
    derivative_rel_tol = 0
    derivative_abs_tol = 5e-6
  []
[]

[Tensors]
  [rho]
    type = Python
    expr = 'Scalar([1.0, 4.0, 9.0, 16.0]).sub_batch.retag(1)'
  []
  [hmat]
    type = Python
    expr = 'Scalar(torch.eye(4, dtype=torch.float64) + 0.5 * (1.0 - torch.eye(4, dtype=torch.float64))).sub_batch.retag(2)'
  []
  [tau]
    type = Python
    expr = 'Scalar(5.0 + 10.0 * torch.sqrt(0.5 * torch.tensor([1.0, 4.0, 9.0, 16.0], dtype=torch.float64) + 15.0)).sub_batch.retag(1)'
  []
[]

[Models]
  [model]
    type = DislocationInteractionStrengthMap
    constant_strength = 5.0
    alpha = 0.5
    mu = 80.0
    b = 0.25
    interaction_matrix = 'hmat'
  []
[]
