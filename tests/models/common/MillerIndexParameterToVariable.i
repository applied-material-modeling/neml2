# MillerIndex-valued ParameterToVariable smoke test (no C++ Catch2 fixture
# exists for this template instantiation; pattern mirrors
# ScalarParameterToVariable.i). Promotes ``from`` to a runtime input so the
# harness has a JVP to check.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_MillerIndex_names = 'param'
    input_MillerIndex_values = 'T'
    output_MillerIndex_names = 'a'
    output_MillerIndex_values = 'T'
  []
[]

[Models]
  [model]
    type = MillerIndexParameterToVariable
    from = 'param'
    to = 'a'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'MillerIndex(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64))'
  []
[]
