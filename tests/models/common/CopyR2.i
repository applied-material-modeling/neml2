# R2-valued CopyVariable smoke test (no C++ Catch2 fixture exists for this
# template instantiation; pattern mirrors R2ConstantParameter.i).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'copy'
    input_R2_names = 'src'
    input_R2_values = 'T'
    output_R2_names = 'dst'
    output_R2_values = 'T'
  []
[]

[Models]
  [copy]
    type = CopyR2
    from = 'src'
    to = 'dst'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'R2(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float64))'
  []
[]
