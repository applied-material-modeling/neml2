# MRP-valued ConstantParameter smoke test (no C++ Catch2 fixture exists for
# this template instantiation; pattern mirrors ScalarConstantParameter.i).
# Promotes ``value`` to a runtime input so the harness has a JVP to check.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_MRP_names = 'value'
    input_MRP_values = 'T'
    output_MRP_names = 'E'
    output_MRP_values = 'T'
  []
[]

[Models]
  [E]
    type = RotConstantParameter
    value = 'value'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'MRP(torch.tensor([0.13991834, 0.18234513, 0.85043991], dtype=torch.float64))'
  []
[]
