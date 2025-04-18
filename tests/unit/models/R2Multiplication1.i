[Tensors]
  [A]
    type = FillR2
    values = '1 3 5 3 5 7 5 7 9'
  []
  [B]
    type = FillR2
    values = '-1 -2 -3 2 5 6 1 3 2'
  []
  [C]
    type = FillR2
    values = '10 28 25 14 40 35 18 52 45'
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
  []
[]

[Models]
  [model]
    type = R2Multiplication
    A = 'state/A'
    B = 'state/B'
    to = 'state/C'
  []
[]
