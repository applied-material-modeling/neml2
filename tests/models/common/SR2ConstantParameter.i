# Translated from tests/unit/models/common/SR2ConstantParameter.i
# (FillSR2 6-value -> SR2.fill; shear slots scaled by sqrt(2) into Mandel).
# The native ModelUnitTest requires at least one JVP comparison, so promote
# ``value`` to a runtime input (mode-4 ``declare_typed_parameter``) by giving
# it a bare variable name; the driver then supplies it via input_SR2_values.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_SR2_names = 'value'
    input_SR2_values = 'T'
    output_SR2_names = 'E'
    output_SR2_values = 'T'
  []
[]

[Models]
  [E]
    type = SR2ConstantParameter
    value = 'value'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'SR2.fill(-1, -4, 7, -1, 9, 1)'
  []
[]
