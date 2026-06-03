# Rot-valued ParameterToVariable smoke test (no C++ Catch2 fixture exists for
# this template instantiation; pattern mirrors ScalarParameterToVariable.i).
# Promotes ``from`` to a runtime input so the harness has a JVP to check.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Rot_names = 'param'
    input_Rot_values = 'T'
    output_Rot_names = 'a'
    output_Rot_values = 'T'
  []
[]

[Models]
  [model]
    type = RotParameterToVariable
    from = 'param'
    to = 'a'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'Rot(torch.tensor([0.13991834, 0.18234513, 0.85043991], dtype=torch.float64))'
  []
[]
