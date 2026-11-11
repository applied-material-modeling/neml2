# LumpedForestDislocationEvolution with k1 and k2 promoted to model inputs, so
# the promoted-parameter derivative actions are exercised.
#
# rho_dot = (k1*sqrt(rho) - k2*rho) * sum|gamma_dot|, with rho = 4, sum = 0.5:
#   entry 0  k1 = 2, k2 = 0.5:  (4 - 2) * 0.5 = 1.0
#   entry 1  k1 = 3, k2 = 0.25: (6 - 1) * 0.5 = 2.5
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'dislocation_density sum_slip_rates k1_in k2_in'
    input_Scalar_values = 'rho gamma_sum k1_vals k2_vals'
    output_Scalar_names = 'dislocation_density_rate'
    output_Scalar_values = 'rho_dot'
  []
[]

[Models]
  [model]
    type = LumpedForestDislocationEvolution
    k1 = 'k1_in'
    k2 = 'k2_in'
  []
[]

[Tensors]
  [rho]
    type = Python
    expr = "Scalar(torch.full((2,), 4.0, dtype=torch.float64))"
  []
  [gamma_sum]
    type = Python
    expr = "Scalar(torch.full((2,), 0.5, dtype=torch.float64))"
  []
  [k1_vals]
    type = Python
    expr = "Scalar(torch.tensor([2.0, 3.0], dtype=torch.float64))"
  []
  [k2_vals]
    type = Python
    expr = "Scalar(torch.tensor([0.5, 0.25], dtype=torch.float64))"
  []
  [rho_dot]
    type = Python
    expr = "Scalar(torch.tensor([1.0, 2.5], dtype=torch.float64))"
  []
[]
