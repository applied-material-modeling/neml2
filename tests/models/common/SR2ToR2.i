# Translated from tests/unit/models/common/SR2toR2.i.
[Tensors]
  [foo]
    type = Python
    expr = "SR2.fill(1.0, 5.0, 9.0, 7.0, 5.0, 3.0)"
  []
  [bar]
    type = Python
    expr = "R2(torch.tensor([[1.0, 3.0, 5.0], [3.0, 5.0, 7.0], [5.0, 7.0, 9.0]], dtype=torch.float64))"
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'sym'
    input_SR2_values = 'foo'
    output_R2_names = 'full'
    output_R2_values = 'bar'
  []
[]

[Models]
  [model]
    type = SR2ToR2
    input = 'sym'
    output = 'full'
  []
[]
