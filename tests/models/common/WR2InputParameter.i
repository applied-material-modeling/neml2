# WR2-valued InputParameter smoke test (no C++ Catch2 fixture; pattern
# mirrors ScalarInputParameter.i).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'ip'
    input_WR2_names = 'variable'
    input_WR2_values = 'T'
    output_WR2_names = 'parameter'
    output_WR2_values = 'T'
  []
[]

[Models]
  [ip]
    type = WR2InputParameter
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'WR2(torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64))'
  []
[]
