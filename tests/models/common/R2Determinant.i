# Translated from tests/unit/models/common/R2Determinant.i
# (FillR2 9-value -> R2(torch.tensor([[...]])); row-major).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_R2_names = 'F'
    input_R2_values = 'F'
    output_Scalar_names = 'J'
    output_Scalar_values = '3.3'
    derivative_abs_tol = 1e-06
  []
[]

[Tensors]
  [F]
    type = Python
    expr = 'R2(torch.tensor([[0.2, 0.5, 0.3], [1.0, 8.0, 7.0], [6.0, 5.0, 2.0]], dtype=torch.float64))'
  []
[]

[Models]
  [model]
    type = R2Determinant
    input = 'F'
    determinant = 'J'
  []
[]
