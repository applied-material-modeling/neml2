# Translated from tests/unit/models/common/R2Multiplication1.i
# (FillR2 9-value -> R2(torch.tensor([[...]])); row-major).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_R2_names = 'A B'
    input_R2_values = 'A B'
    output_R2_names = 'C'
    output_R2_values = 'C'
  []
[]

[Tensors]
  [A]
    type = Python
    expr = 'R2(torch.tensor([[1.0, 3.0, 5.0], [3.0, 5.0, 7.0], [5.0, 7.0, 9.0]], dtype=torch.float64))'
  []
  [B]
    type = Python
    expr = 'R2(torch.tensor([[-1.0, -2.0, -3.0], [2.0, 5.0, 6.0], [1.0, 3.0, 2.0]], dtype=torch.float64))'
  []
  [C]
    type = Python
    expr = 'R2(torch.tensor([[10.0, 28.0, 25.0], [14.0, 40.0, 35.0], [18.0, 52.0, 45.0]], dtype=torch.float64))'
  []
[]

[Models]
  [model]
    type = R2Multiplication
    A = 'A'
    B = 'B'
    to = 'C'
  []
[]
