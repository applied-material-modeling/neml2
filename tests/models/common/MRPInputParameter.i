# MRP-valued InputParameter smoke test (no C++ Catch2 fixture; pattern
# mirrors ScalarInputParameter.i).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'ip'
    input_MRP_names = 'variable'
    input_MRP_values = 'T'
    output_MRP_names = 'parameter'
    output_MRP_values = 'T'
  []
[]

[Models]
  [ip]
    type = MRPInputParameter
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'MRP(torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64))'
  []
[]
