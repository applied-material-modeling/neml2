# WR2-valued ConstantParameter smoke test (no C++ Catch2 fixture exists for
# this template instantiation; pattern mirrors ScalarConstantParameter.i).
# Promotes ``value`` to a runtime input so the harness has a JVP to check.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_WR2_names = 'value'
    input_WR2_values = 'T'
    output_WR2_names = 'E'
    output_WR2_values = 'T'
  []
[]

[Models]
  [E]
    type = WR2ConstantParameter
    value = 'value'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'WR2(torch.tensor([0.01, -0.02, 0.03], dtype=torch.float64))'
  []
[]
