# Vec-valued InputParameter smoke test (no C++ Catch2 fixture; pattern
# mirrors ScalarInputParameter.i).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'ip'
    input_Vec_names = 'variable'
    input_Vec_values = 'T'
    output_Vec_names = 'parameter'
    output_Vec_values = 'T'
  []
[]

[Models]
  [ip]
    type = VecInputParameter
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'Vec(torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64))'
  []
[]
