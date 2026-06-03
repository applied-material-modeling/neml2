# R2-valued InputParameter smoke test (no C++ Catch2 fixture; pattern
# mirrors ScalarInputParameter.i).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'ip'
    input_R2_names = 'variable'
    input_R2_values = 'T'
    output_R2_names = 'parameter'
    output_R2_values = 'T'
  []
[]

[Models]
  [ip]
    type = R2InputParameter
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'R2(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float64))'
  []
[]
