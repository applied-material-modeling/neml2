# KocksMeckingDynamicRecovery with k20 and gamma0 promoted to model inputs, so
# the promoted-parameter derivative actions are exercised.
#
# k2 = k20 * (max(g, floor) / g0)^(-1/n), floor = 1e-10:
#   entry 0  below floor: g = 1e-12, n = 5,  k20 = 0.5, g0 = 1:   50
#   entry 1  k20 = 0:     g = 1e-3,  n = 10, g0 = 1:              0
#            (d k2 / d k20 = (1e-3)^(-1/10) stays finite)
#   entry 2               g = 0.1,   n = 20, k20 = 2,   g0 = 0.5: 2.16759677347
#   entry 3               g = 2,     n = 8,  k20 = 0.5, g0 = 4:   0.54525386633
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'sum_slip_rates rate_sensitivity k20_in gamma0_in'
    input_Scalar_values = 'g n k20_vals g0_vals'
    output_Scalar_names = 'k2'
    output_Scalar_values = 'k2'
  []
[]

[Models]
  [model]
    type = KocksMeckingDynamicRecovery
    k20 = 'k20_in'
    gamma0 = 'gamma0_in'
  []
[]

[Tensors]
  [g]
    type = Python
    expr = "Scalar(torch.tensor([1e-12, 1e-3, 0.1, 2.0], dtype=torch.float64))"
  []
  [n]
    type = Python
    expr = "Scalar(torch.tensor([5.0, 10.0, 20.0, 8.0], dtype=torch.float64))"
  []
  [k20_vals]
    type = Python
    expr = "Scalar(torch.tensor([0.5, 0.0, 2.0, 0.5], dtype=torch.float64))"
  []
  [g0_vals]
    type = Python
    expr = "Scalar(torch.tensor([1.0, 1.0, 0.5, 4.0], dtype=torch.float64))"
  []
  [k2]
    type = Python
    expr = "Scalar(torch.tensor([50.0, 0.0, 2.16759677347, 0.54525386633], dtype=torch.float64))"
  []
[]
