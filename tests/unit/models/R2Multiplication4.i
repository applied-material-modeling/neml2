[Tensors]
  [A]
    type = FillR2
    values = '-1 -2 -3 2 5 6 1 3 2'
  []
  [B]
    type = FillR2
    values = '40 53 71 53 88 113 71 113 160'
  []
  [C]
    type = FillR2
    values = '-0.91353001 -0.55035605 0.8128179 0.22024415 0.04526958 -0.12970498 0.11322482 0.16876907 -0.17568667'
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
    derivative_abs_tol = 1e-4
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
    invert_B = true
  []
[]
