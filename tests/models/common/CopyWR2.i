# WR2-valued CopyVariable smoke test (no C++ Catch2 fixture exists for this
# template instantiation; pattern mirrors WR2ConstantParameter.i).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'copy'
    input_WR2_names = 'src'
    input_WR2_values = 'T'
    output_WR2_names = 'dst'
    output_WR2_values = 'T'
  []
[]

[Models]
  [copy]
    type = CopyWR2
    from = 'src'
    to = 'dst'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'WR2(torch.tensor([0.01, -0.02, 0.03], dtype=torch.float64))'
  []
[]
