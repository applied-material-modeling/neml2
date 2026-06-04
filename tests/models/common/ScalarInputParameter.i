# Scalar-valued InputParameter smoke test. The C++ side has no .i Catch2
# fixture because the model "is not intended to be used directly in the input
# file" (it's normally synthesized when a [Models] output wires to a [Parameters]
# slot). The native ModelUnitTest still drives it cleanly: feed ``variable`` and
# expect ``parameter`` to come out identically, with the identity JVP.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'ip'
    input_Scalar_names = 'variable'
    input_Scalar_values = 'T'
    output_Scalar_names = 'parameter'
    output_Scalar_values = 'T'
  []
[]

[Models]
  [ip]
    type = ScalarInputParameter
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'Scalar(2.5)'
  []
[]
