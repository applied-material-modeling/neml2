# MillerIndex-valued CopyVariable smoke test (no C++ Catch2 fixture exists for
# this template instantiation; pattern mirrors MillerIndexConstantParameter.i).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'copy'
    input_MillerIndex_names = 'src'
    input_MillerIndex_values = 'T'
    output_MillerIndex_names = 'dst'
    output_MillerIndex_values = 'T'
  []
[]

[Models]
  [copy]
    type = CopyMillerIndex
    from = 'src'
    to = 'dst'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'MillerIndex(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64))'
  []
[]
