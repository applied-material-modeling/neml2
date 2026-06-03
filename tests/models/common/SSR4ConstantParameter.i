# SSR4-valued ConstantParameter smoke test (no C++ Catch2 fixture exists for
# this template instantiation; pattern mirrors ScalarConstantParameter.i).
# Promotes ``value`` to a runtime input so the harness has a JVP to check.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_SSR4_names = 'value'
    input_SSR4_values = 'T'
    output_SSR4_names = 'E'
    output_SSR4_values = 'T'
  []
[]

[Models]
  [E]
    type = SSR4ConstantParameter
    value = 'value'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'SSR4(torch.eye(6, dtype=torch.float64))'
  []
[]
