# Rot-valued ConstantParameter smoke test (no C++ Catch2 fixture exists for
# this template instantiation; pattern mirrors ScalarConstantParameter.i).
# Promotes ``value`` to a runtime input so the harness has a JVP to check.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_Rot_names = 'value'
    input_Rot_values = 'T'
    output_Rot_names = 'E'
    output_Rot_values = 'T'
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
    expr = 'Rot(torch.tensor([0.13991834, 0.18234513, 0.85043991], dtype=torch.float64))'
  []
[]
