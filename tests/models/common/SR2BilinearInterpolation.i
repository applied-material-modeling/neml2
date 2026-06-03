# Mirrors ScalarBilinearInterpolation1.i with an SR2 ordinate.
#
#   abscissa1: sub_batch (N1=3,)
#   abscissa2: sub_batch (N2=2,)
#    ordinate: sub_batch (3, 2,) + base (6,) — each cell's six Mandel components
#       equal the scalar test's value at that cell, so the (T, eps) = (0.5, 0.5)
#       query interpolates to 2.25 on every component.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_Scalar_names = 'T eps'
    input_Scalar_values = '0.5 0.5'
    output_SR2_names = 'E'
    output_SR2_values = 'E_correct'
  []
[]

[Models]
  [E]
    type = SR2BilinearInterpolation
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
    expr = "Scalar(torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)).with_sub_batch(1)"
  []
  [eps_vals]
    type = Python
    expr = "Scalar(torch.tensor([0.0, 2.0], dtype=torch.float64)).with_sub_batch(1)"
  []
  [S]
    type = Python
    expr = "SR2(torch.tensor([[[1.0]*6, [2.0]*6], [[3.0]*6, [4.0]*6], [[5.0]*6, [6.0]*6]], dtype=torch.float64)).with_sub_batch(2)"
  []
  [E_correct]
    type = Python
    expr = "SR2(torch.tensor([2.25]*6, dtype=torch.float64))"
  []
[]
