# Translated from tests/unit/models/solid_mechanics/crystal_plasticity/FixOrientation.i.
# The C++ fixture's FillRot 'x y z' maps to Rot(torch.tensor([x, y, z])).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Rot_names = 'input'
    input_Rot_values = 'R_in'
    output_Rot_names = 'output'
    output_Rot_values = 'R_out'
  []
[]

[Tensors]
  [R_in]
    type = Python
    expr = 'Rot(torch.tensor([1.0, -0.1, -0.05], dtype=torch.float64))'
  []
  [R_out]
    type = Python
    expr = 'Rot(torch.tensor([-0.98765432, 0.09876543, 0.04938272], dtype=torch.float64))'
  []
[]

[Models]
  [model]
    type = FixOrientation
  []
[]
