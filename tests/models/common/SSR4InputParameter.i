# SSR4-valued InputParameter smoke test (no C++ Catch2 fixture; pattern
# mirrors ScalarInputParameter.i).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'ip'
    input_SSR4_names = 'variable'
    input_SSR4_values = 'T'
    output_SSR4_names = 'parameter'
    output_SSR4_values = 'T'
  []
[]

[Models]
  [ip]
    type = SSR4InputParameter
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'SSR4(torch.eye(6, dtype=torch.float64))'
  []
[]
