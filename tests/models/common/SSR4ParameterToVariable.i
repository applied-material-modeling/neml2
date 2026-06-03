# SSR4-valued ParameterToVariable smoke test (no C++ Catch2 fixture exists for
# this template instantiation; pattern mirrors ScalarParameterToVariable.i).
# Promotes ``from`` to a runtime input so the harness has a JVP to check.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SSR4_names = 'param'
    input_SSR4_values = 'T'
    output_SSR4_names = 'a'
    output_SSR4_values = 'T'
  []
[]

[Models]
  [model]
    type = SSR4ParameterToVariable
    from = 'param'
    to = 'a'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'SSR4(torch.eye(6, dtype=torch.float64))'
  []
[]
