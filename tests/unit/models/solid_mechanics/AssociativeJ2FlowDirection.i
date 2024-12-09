[Tensors]
  [M]
    type = FillSR2
    values = '1 2 3 4 5 6'
  []
  [NM]
    type = FillSR2
    values = '-0.09805807 0 0.09805807 0.39223227 0.49029034 0.58834841'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_symr2_names = 'state/M'
    input_symr2_values = 'M'
    output_symr2_names = 'state/NM'
    output_symr2_values = 'NM'
    derivative_abs_tol = 1e-6
  []
[]

[Models]
  [model]
    type = AssociativeJ2FlowDirection
    mandel_stress = 'state/M'
    flow_direction = 'state/NM'
  []
[]
