# Rot-valued InputParameter smoke test (no C++ Catch2 fixture; pattern
# mirrors ScalarInputParameter.i).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'ip'
    input_Rot_names = 'variable'
    input_Rot_values = 'T'
    output_Rot_names = 'parameter'
    output_Rot_values = 'T'
  []
[]

[Models]
  [ip]
    type = RotInputParameter
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'Rot(torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64))'
  []
[]
