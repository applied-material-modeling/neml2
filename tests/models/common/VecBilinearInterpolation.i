# Mirrors ScalarBilinearInterpolation1.i with a Vec ordinate.
#
#   abscissa1: sub_batch (N1=3,)
#   abscissa2: sub_batch (N2=2,)
#    ordinate: sub_batch (3, 2,) + base (3,) — Y[i,j,k] = scalar_test[i,j] * 10**k
#       at (T, eps) = (0.5, 0.5) the scalar test interpolates to 2.25, so
#       expected Vec is [2.25, 22.5, 225].
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_Scalar_names = 'T eps'
    input_Scalar_values = '0.5 0.5'
    output_Vec_names = 'E'
    output_Vec_values = 'E_correct'
  []
[]

[Models]
  [E]
    type = VecBilinearInterpolation
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
    expr = "Vec(torch.tensor([[[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]], [[3.0, 30.0, 300.0], [4.0, 40.0, 400.0]], [[5.0, 50.0, 500.0], [6.0, 60.0, 600.0]]], dtype=torch.float64)).sub_batch.retag(2)"
  []
  [E_correct]
    type = Python
    expr = "Vec(torch.tensor([2.25, 22.5, 225.0], dtype=torch.float64))"
  []
[]
