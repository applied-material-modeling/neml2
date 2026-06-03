# Vec-valued ParameterToVariable smoke test (no C++ Catch2 fixture exists for
# this template instantiation; pattern mirrors ScalarParameterToVariable.i).
# Promotes ``from`` to a runtime input so the harness has a JVP to check.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Vec_names = 'param'
    input_Vec_values = 'T'
    output_Vec_names = 'a'
    output_Vec_values = 'T'
  []
[]

[Models]
  [model]
    type = VecParameterToVariable
    from = 'param'
    to = 'a'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'Vec(torch.tensor([1.0, -2.0, 3.5], dtype=torch.float64))'
  []
[]
