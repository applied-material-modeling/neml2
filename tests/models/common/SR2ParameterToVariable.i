# SR2-valued ParameterToVariable smoke test (no C++ Catch2 fixture exists for
# this template instantiation; pattern mirrors ScalarParameterToVariable.i).
# Promotes ``from`` to a runtime input so the harness has a JVP to check.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'param'
    input_SR2_values = 'T'
    output_SR2_names = 'a'
    output_SR2_values = 'T'
  []
[]

[Models]
  [model]
    type = SR2ParameterToVariable
    from = 'param'
    to = 'a'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'SR2.fill(-1, -4, 7, -1, 9, 1)'
  []
[]
