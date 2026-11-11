# y = (x^-p + xmax^-p)^(-1/p) with xmax = 50, p = 4, sweeping x across the cap:
#   x = 5     -> 4.99987500781  (far below: y ~ x)
#   x = 10    -> 9.99600399520  (0.04% below x)
#   x = 50    -> 42.0448207627  (at the cap: 2^(-1/4) * xmax)
#   x = 100   -> 49.2479060505
#   x = 500   -> 49.9987500781  (far above: y ~ xmax)
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'x'
    input_Scalar_values = 'x'
    output_Scalar_names = 'y'
    output_Scalar_values = 'y'
  []
[]

[Tensors]
  [x]
    type = Python
    expr = "Scalar(torch.tensor([5.0, 10.0, 50.0, 100.0, 500.0], dtype=torch.float64))"
  []
  [y]
    type = Python
    expr = "Scalar(torch.tensor([4.9998750078, 9.9960039952, 42.0448207627, 49.2479060505, 49.9987500781], dtype=torch.float64))"
  []
[]

[Models]
  [model]
    type = SmoothCap
    argument = 'x'
    value = 'y'
    maximum = 50.0
    exponent = 4.0
  []
[]
