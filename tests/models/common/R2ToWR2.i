# Translated from tests/unit/models/common/R2ToWR2.i.
[Tensors]
  [foo]
    type = Python
    expr = "R2(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float64))"
  []
  [bar]
    type = Python
    expr = "WR2(torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64))"
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_R2_names = 'full'
    input_R2_values = 'foo'
    output_WR2_names = 'notfull'
    output_WR2_values = 'bar'
  []
[]

[Models]
  [model]
    type = R2ToWR2
    input = 'full'
    output = 'notfull'
  []
[]
