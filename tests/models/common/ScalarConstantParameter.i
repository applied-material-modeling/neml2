# Translated from tests/unit/models/common/ScalarConstantParameter.i.
# The native ModelUnitTest requires at least one JVP comparison, so promote
# ``value`` to a runtime input (mode-4 ``declare_typed_parameter``) by giving
# it a bare variable name; the driver then supplies it via input_Scalar_values.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_Scalar_names = 'value'
    input_Scalar_values = 'T'
    output_Scalar_names = 'E'
    output_Scalar_values = 'T'
  []
[]

[Models]
  [E]
    type = ScalarConstantParameter
    value = 'value'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'Scalar(20.0)'
  []
[]
