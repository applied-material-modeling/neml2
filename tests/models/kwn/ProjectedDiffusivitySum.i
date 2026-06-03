# Translated from tests/unit/models/kwn/ProjectedDiffusivitySum.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/x1 state/x2'
    input_Scalar_values = 'x1 x2'
    output_Scalar_names = 'state/sum'
    output_Scalar_values = 'sum'
  []
[]

[Tensors]
  [dx1]
    type = Python
    expr = "Scalar(torch.tensor([0.2, 0.1], dtype=torch.float64))"
  []
  [dx2]
    type = Python
    expr = "Scalar(torch.tensor([0.05, 0.15], dtype=torch.float64))"
  []
  [x1]
    type = Python
    expr = "Scalar(torch.tensor([0.4, 0.6], dtype=torch.float64))"
  []
  [x2]
    type = Python
    expr = "Scalar(torch.tensor([0.8, 0.5], dtype=torch.float64))"
  []
  [sum]
    type = Python
    expr = "Scalar(torch.tensor([0.215625, 0.25833333333333336], dtype=torch.float64))"
  []
[]

[Models]
  [model]
    type = ProjectedDiffusivitySum
    concentration_differences = 'dx1 dx2'
    diffusivities = '0.5 0.2'
    far_field_concentrations = 'state/x1 state/x2'
    projected_diffusivity_sum = 'state/sum'
  []
[]
