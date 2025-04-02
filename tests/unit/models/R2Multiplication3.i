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
    values = '0.03865717 0.00101729 -0.03662258 -0.12380468 0.13621567 -0.00376399 -0.05879959 0.21424212 -0.11271617'
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
    invert_B = true
  []
[]
