# k2 = k20 * (max(g, floor) / g0)^(-1/n) with k20 = 0.5, g0 = 1, floor = 1e-10:
#   entry 0  g = 1e-12 (below floor), n = 5:  0.5 * (1e-10)^(-1/5) = 50
#   entry 1  g = 1e-3,                n = 10: 0.5 * (1e-3)^(-1/10)  = 0.99763115748
#   entry 2  g = 0.1,                 n = 20: 0.5 * (0.1)^(-1/20)   = 0.56100922715
#   entry 3  g = 2,                   n = 8:  0.5 * 2^(-1/8)        = 0.45850202160
# Below the floor the slip-rate derivative is zero.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'sum_slip_rates rate_sensitivity'
    input_Scalar_values = 'g n'
    output_Scalar_names = 'k2'
    output_Scalar_values = 'k2'
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
  [k2]
    type = Python
    expr = "Scalar(torch.tensor([50.0, 0.99763115748, 0.56100922715, 0.45850202160], dtype=torch.float64))"
  []
[]

[Models]
  [model]
    type = KocksMeckingDynamicRecovery
    k20 = 0.5
    gamma0 = 1.0
  []
[]
