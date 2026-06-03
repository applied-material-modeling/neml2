# Translated from tests/unit/models/solid_mechanics/plasticity/AssociativeJ2FlowDirection.i
# (FillSR2 6-value -> SR2.fill; shear slots scaled by sqrt(2)).
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

[Tensors]
  [M]
    type = Python
    expr = 'SR2.fill(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)'
  []
  [NM]
    type = Python
    expr = 'SR2.fill(-0.09805807, 0.0, 0.09805807, 0.39223227, 0.49029034, 0.58834841)'
  []
[]

[Models]
  [model]
    type = AssociativeJ2FlowDirection
  []
[]
