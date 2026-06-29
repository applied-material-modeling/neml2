# Translated from tests/unit/models/solid_mechanics/crystal_plasticity/FixOrientation.i.
# The C++ fixture's FillRot 'x y z' maps to MRP(torch.tensor([x, y, z])).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_MRP_names = 'input'
    input_MRP_values = 'R_in'
    output_MRP_names = 'output'
    output_MRP_values = 'R_out'
  []
[]

[Tensors]
  [R_in]
    type = Python
    expr = 'MRP(torch.tensor([1.0, -0.1, -0.05], dtype=torch.float64))'
  []
  [R_out]
    type = Python
    expr = 'MRP(torch.tensor([-0.98765432, 0.09876543, 0.04938272], dtype=torch.float64))'
  []
[]

[Models]
  [model]
    type = FixOrientation
  []
[]
