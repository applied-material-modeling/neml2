# R2-valued ConstantParameter smoke test (no C++ Catch2 fixture exists for
# this template instantiation; pattern mirrors ScalarConstantParameter.i).
# Promotes ``value`` to a runtime input so the harness has a JVP to check.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_R2_names = 'value'
    input_R2_values = 'T'
    output_R2_names = 'E'
    output_R2_values = 'T'
  []
[]

[Models]
  [E]
    type = R2ConstantParameter
    value = 'value'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'R2(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float64))'
  []
[]
