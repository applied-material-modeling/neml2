# WR2-valued ParameterToVariable smoke test (no C++ Catch2 fixture exists for
# this template instantiation; pattern mirrors ScalarParameterToVariable.i).
# Promotes ``from`` to a runtime input so the harness has a JVP to check.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_WR2_names = 'param'
    input_WR2_values = 'T'
    output_WR2_names = 'a'
    output_WR2_values = 'T'
  []
[]

[Models]
  [model]
    type = WR2ParameterToVariable
    from = 'param'
    to = 'a'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'WR2(torch.tensor([0.01, -0.02, 0.03], dtype=torch.float64))'
  []
[]
