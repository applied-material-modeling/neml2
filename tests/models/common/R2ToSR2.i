# Translated from tests/unit/models/common/R2ToSR2.i.
[Tensors]
  [foo]
    type = Python
    expr = "R2(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float64))"
  []
  [bar]
    type = Python
    expr = "SR2.fill(1.0, 5.0, 9.0, 7.0, 5.0, 3.0)"
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_R2_names = 'full'
    input_R2_values = 'foo'
    output_SR2_names = 'notfull'
    output_SR2_values = 'bar'
  []
[]

[Models]
  [model]
    type = R2ToSR2
    input = 'full'
    output = 'notfull'
  []
[]
