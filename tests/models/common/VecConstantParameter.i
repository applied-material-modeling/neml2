# Vec-valued ConstantParameter smoke test (no C++ Catch2 fixture exists for
# this template instantiation; pattern mirrors ScalarConstantParameter.i).
# Promotes ``value`` to a runtime input so the harness has a JVP to check.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_Vec_names = 'value'
    input_Vec_values = 'T'
    output_Vec_names = 'E'
    output_Vec_values = 'T'
  []
[]

[Models]
  [E]
    type = VecConstantParameter
    value = 'value'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'Vec(torch.tensor([1.0, -2.0, 3.5], dtype=torch.float64))'
  []
[]
