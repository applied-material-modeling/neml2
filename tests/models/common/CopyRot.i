# Rot-valued CopyVariable smoke test (no C++ Catch2 fixture exists for this
# template instantiation; pattern mirrors RotConstantParameter.i).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'copy'
    input_Rot_names = 'src'
    input_Rot_values = 'T'
    output_Rot_names = 'dst'
    output_Rot_values = 'T'
  []
[]

[Models]
  [copy]
    type = CopyRot
    from = 'src'
    to = 'dst'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'Rot(torch.tensor([0.13991834, 0.18234513, 0.85043991], dtype=torch.float64))'
  []
[]
