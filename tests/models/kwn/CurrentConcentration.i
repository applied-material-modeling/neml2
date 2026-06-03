# Translated from tests/unit/models/kwn/CurrentConcentration.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/f1 state/f2'
    input_Scalar_values = 'f1 f2'
    output_Scalar_names = 'state/x'
    output_Scalar_values = 'x'
  []
[]

[Tensors]
  [f1]
    type = Python
    expr = "Scalar(torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64))"
  []
  [f2]
    type = Python
    expr = "Scalar(torch.tensor([0.05, 0.1, 0.2], dtype=torch.float64))"
  []
  [x]
    type = Python
    expr = "Scalar(torch.tensor([0.16470588235294118, 0.04285714285714286, -0.22], dtype=torch.float64))"
  []
[]

[Models]
  [model]
    type = CurrentConcentration
    initial_concentration = 0.25
    precipitate_volume_fractions = 'state/f1 state/f2'
    precipitate_concentrations = '0.8 0.6'
    current_concentration = 'state/x'
  []
[]
