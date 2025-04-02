[Tensors]
  [A]
    type = FillR2
    values = '40 53 71 53 88 113 71 113 160'
  []
  [B]
    type = FillR2
    values = '-1 -2 -3 2 5 6 1 3 2'
  []
  [C]
    type = FillR2
    values = '-0.25269583 -0.57822991 -0.73173957 0.24598169 0.55320448 0.81495422 -0.05534079 -0.11536114 -0.23835198'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_R2_names = 'state/A state/B'
    input_R2_values = 'A B'
    output_R2_names = 'state/C'
    output_R2_values = 'C'
    derivative_abs_tol = 1e-5
    derivative_rel_tol = 0
  []
[]

[Models]
  [model]
    type = R2Multiplication
    A = 'state/A'
    B = 'state/B'
    to = 'state/C'
    invert_A = true
  []
[]
