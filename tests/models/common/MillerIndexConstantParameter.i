# MillerIndex-valued ConstantParameter smoke test (no C++ Catch2 fixture exists
# for this template instantiation; pattern mirrors ScalarConstantParameter.i).
# Promotes ``value`` to a runtime input so the harness has a JVP to check.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_MillerIndex_names = 'value'
    input_MillerIndex_values = 'T'
    output_MillerIndex_names = 'E'
    output_MillerIndex_values = 'T'
  []
[]

[Models]
  [E]
    type = MillerIndexConstantParameter
    value = 'value'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'MillerIndex(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64))'
  []
[]
