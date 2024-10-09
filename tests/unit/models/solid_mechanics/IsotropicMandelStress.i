[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_symr2_names = 'state/S'
    input_symr2_values = 'S'
    output_symr2_names = 'state/internal/M'
    output_symr2_values = 'M'
  []
[]

[Tensors]
  [S]
    type = FillSR2
    values = '50 -10 20 40 30 -60'
  []
  [M]
    type = FillSR2
    values = '50 -10 20 40 30 -60'
  []
[]

[Models]
  [model]
    type = IsotropicMandelStress
  []
[]
