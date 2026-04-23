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
    input_SR2_names = 'mandel_stress'
    input_SR2_values = 'M'
    output_SR2_names = 'flow_direction'
    output_SR2_values = 'NM'
    derivative_abs_tol = 1e-6
  []
[]

[Models]
  [model]
    type = AssociativeJ2FlowDirection
    mandel_stress = 'mandel_stress'
    flow_direction = 'flow_direction'
  []
[]
