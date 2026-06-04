# Translated from tests/unit/models/common/ParameterToVariable.i.
# The native ModelUnitTest requires at least one JVP comparison, so promote
# ``from`` to a runtime input (mode-4 ``declare_typed_parameter``) by giving
# it a bare variable name; the driver then supplies it via input_Scalar_values.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'param'
    input_Scalar_values = 'T'
    output_Scalar_names = 'a'
    output_Scalar_values = 'T'
  []
[]

[Models]
  [model]
    type = ScalarParameterToVariable
    from = 'param'
    to = 'a'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'Scalar(1.61753845)'
  []
[]
