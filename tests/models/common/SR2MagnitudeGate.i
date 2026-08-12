# SR2 variant of the elementwise selector, gating on the Frobenius norm.
#
# Member 0 is cold (zero reference) and takes the prediction; member 1 is warm
# and passes its reference through untouched.
[Tensors]
  [prediction]
    type = Python
    expr = 'SR2(torch.tensor([[0.5, -0.25, 0.125, 0.0625, -0.03125, 0.25], [0.5, -0.25, 0.125, 0.0625, -0.03125, 0.25]], dtype=torch.float64))'
  []
  [reference]
    type = Python
    expr = 'SR2(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.5, -0.5, 0.25, 0.125, -0.0625, 0.5]], dtype=torch.float64))'
  []
  [gated]
    type = Python
    expr = 'SR2(torch.tensor([[0.5, -0.25, 0.125, 0.0625, -0.03125, 0.25], [1.5, -0.5, 0.25, 0.125, -0.0625, 0.5]], dtype=torch.float64))'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'prediction reference'
    input_SR2_values = 'prediction reference'
    output_SR2_names = 'gated'
    output_SR2_values = 'gated'
  []
[]

[Models]
  [model]
    type = SR2MagnitudeGate
    threshold = 1e-3
  []
[]
