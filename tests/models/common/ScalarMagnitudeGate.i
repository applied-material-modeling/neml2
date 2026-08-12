# Elementwise selector: y = ||reference|| < threshold ? prediction : reference.
#
# The batch deliberately straddles the threshold so both branches are exercised
# in one call and the JVP check sees the tangent routed each way. Values sit
# well clear of the threshold: the switch is a jump there, so a finite
# difference across it would be meaningless.
[Tensors]
  [prediction]
    type = Python
    expr = 'Scalar(torch.tensor([7.0, 7.0, -3.5], dtype=torch.float64))'
  []
  # |ref| = 0 (cold), 42 (warm), 4.2e-8 (cold, and negative -- coldness is on
  # the magnitude, not the sign).
  [reference]
    type = Python
    expr = 'Scalar(torch.tensor([0.0, 42.0, -4.2e-8], dtype=torch.float64))'
  []
  [gated]
    type = Python
    expr = 'Scalar(torch.tensor([7.0, 42.0, -3.5], dtype=torch.float64))'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'prediction reference'
    input_Scalar_values = 'prediction reference'
    output_Scalar_names = 'gated'
    output_Scalar_values = 'gated'
  []
[]

[Models]
  [model]
    type = ScalarMagnitudeGate
    threshold = 1e-3
  []
[]
