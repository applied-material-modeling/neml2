# Translated from tests/unit/models/common/SR2LinearCombination.i (FillSR2 6-value -> SR2.fill).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'A B'
    input_SR2_values = 'foo bar'
    output_SR2_names = 'C'
    output_SR2_values = 'baz'
  []
[]

[Tensors]
  [foo]
    type = Python
    expr = 'SR2.fill(1, 2, 3, 4, 5, 6)'
  []
  [bar]
    type = Python
    expr = 'SR2.fill(-1, -4, 7, -1, 9, 1)'
  []
  [baz]
    type = Python
    expr = 'SR2.fill(0, -2, 10, 3, 14, 7)'
  []
[]

[Models]
  [model]
    type = SR2LinearCombination
    from = 'A B'
    to = 'C'
  []
[]
