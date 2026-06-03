# Translated from tests/unit/models/common/ScalarToDiagonalSR2.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'input'
    input_Scalar_values = 'input'
    output_SR2_names = 'output'
    output_SR2_values = 'output'
  []
[]

[Tensors]
  [input]
    type = Python
    expr = 'Scalar(2.1)'
  []
  [output]
    type = Python
    expr = 'SR2.fill(2.1, 2.1, 2.1, 0.0, 0.0, 0.0)'
  []
[]

[Models]
  [model]
    type = ScalarToDiagonalSR2
    input = 'input'
    output = 'output'
  []
[]
