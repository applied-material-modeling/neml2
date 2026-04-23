[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Rot_names = 'input'
    input_Rot_values = 'R_in'
    output_Rot_names = 'output'
    output_Rot_values = 'R_out'
  []
[]

[Tensors]
  [R_in]
    type = FillRot
    values = '1.0 -0.1 -0.05'
  []
  [R_out]
    type = FillRot
    values = '-0.98765432 0.09876543 0.04938272'
  []
[]

[Models]
  [model]
    type = FixOrientation
    input = 'input'
    output = 'output'
  []
[]
