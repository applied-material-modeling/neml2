# Model unit test: DislocationInteractionStrengthMap with a PROMOTED interaction matrix.
# tau_i = constant_strength + alpha*mu*b*sqrt(sum_r h_ir rho_r)
#
# Promoting interaction_matrix to a runtime input (mode-4: a bare variable name with
# no matching [Tensors] entry) makes ModelUnitTest's JVP check cover the analytic
# d(tau)/d(h) chain-rule action -- the derivative path used when the matrix is
# calibrated. The companion DislocationInteractionStrengthMap.i keeps the matrix a
# static parameter (checks d(tau)/d(rho) only).
#
# Same numbers as that test: rho=(1,4,9,16), h = I + 0.5(J-I) (so the forest sum is
# 0.5*rho_i + 15), constant_strength=5, alpha=0.5, mu=80, b=0.25 -> tau_i = 5 +
# 10*sqrt(0.5*rho_i + 15).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'dislocation_density h_in'
    input_Scalar_values = 'rho hmat'
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
    interaction_matrix = 'h_in'
  []
[]
