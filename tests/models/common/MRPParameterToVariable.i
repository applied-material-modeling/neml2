# MRP-valued ParameterToVariable smoke test (no C++ Catch2 fixture exists for
# this template instantiation; pattern mirrors ScalarParameterToVariable.i).
# Promotes ``from`` to a runtime input so the harness has a JVP to check.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_MRP_names = 'param'
    input_MRP_values = 'T'
    output_MRP_names = 'a'
    output_MRP_values = 'T'
  []
[]

[Models]
  [model]
    type = MRPParameterToVariable
    from = 'param'
    to = 'a'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'MRP(torch.tensor([0.13991834, 0.18234513, 0.85043991], dtype=torch.float64))'
  []
[]
