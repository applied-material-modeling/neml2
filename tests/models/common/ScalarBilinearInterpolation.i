# Translated from tests/unit/models/common/ScalarBilinearInterpolation1.i.
#
# Unbatched 2-D table:
#   abscissa1: sub_batch (N1=3,)
#   abscissa2: sub_batch (N2=2,)
#    ordinate: sub_batch (3, 2,)
#   argument1: scalar query
#   argument2: scalar query
#      output: scalar value at (T, eps)
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_Scalar_names = 'T eps'
    input_Scalar_values = '0.5 0.5'
    output_Scalar_names = 'E'
    output_Scalar_values = '2.25'
  []
[]

[Models]
  [E]
    type = ScalarBilinearInterpolation
    argument1 = 'T'
    argument2 = 'eps'
    abscissa1 = 'T_vals'
    abscissa2 = 'eps_vals'
    ordinate = 'S'
  []
[]

[Tensors]
  [T_vals]
    type = Python
    expr = "Scalar(torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)).sub_batch.retag(1)"
  []
  [eps_vals]
    type = Python
    expr = "Scalar(torch.tensor([0.0, 2.0], dtype=torch.float64)).sub_batch.retag(1)"
  []
  [S]
    type = Python
    expr = "Scalar(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)).sub_batch.retag(2)"
  []
[]
