# MillerIndex-valued InputParameter smoke test (no C++ Catch2 fixture;
# pattern mirrors ScalarInputParameter.i).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'ip'
    input_MillerIndex_names = 'variable'
    input_MillerIndex_values = 'T'
    output_MillerIndex_names = 'parameter'
    output_MillerIndex_values = 'T'
  []
[]

[Models]
  [ip]
    type = MillerIndexInputParameter
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'MillerIndex(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64))'
  []
[]
