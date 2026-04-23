[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'elastic_strain plastic_deformation_rate'
    input_SR2_values = 'e dp'
    input_WR2_names = 'vorticity plastic_vorticity'
    input_WR2_values = 'w wp'
    output_WR2_names = 'orientation_rate'
    output_WR2_values = 'r_rate'
  []
[]

[Tensors]
  [e]
    type = FillSR2
    values = '0.100 0.110 0.100 0.050 0.040 0.030'
  []
  [dp]
    type = FillSR2
    values = '0.050 0.120 0.080 0.010 0.010 0.090'
  []
  [w]
    type = FillWR2
    values = '0.01 -0.02 0.03'
  []
  [wp]
    type = FillWR2
    values = '0.03 0.02 -0.01'
  []
  [r_rate]
    type = FillWR2
    values = '-0.0252 -0.037  0.0411'
  []
[]

[Models]
  [model]
    type = OrientationRate
  []
[]
