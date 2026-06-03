# Vec-valued CopyVariable smoke test (no C++ Catch2 fixture exists for this
# template instantiation; pattern mirrors VecConstantParameter.i).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'copy'
    input_Vec_names = 'src'
    input_Vec_values = 'T'
    output_Vec_names = 'dst'
    output_Vec_values = 'T'
  []
[]

[Models]
  [copy]
    type = CopyVec
    from = 'src'
    to = 'dst'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'Vec(torch.tensor([1.0, -2.0, 3.5], dtype=torch.float64))'
  []
[]
