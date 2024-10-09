[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_symr2_names = 'state/elastic_strain state/internal/plastic_deformation_rate'
    input_symr2_values = 'e dp'
    input_skewr2_names = 'forces/vorticity state/internal/plastic_vorticity'
    input_skewr2_values = 'w wp'
    output_skewr2_names = 'state/orientation_rate'
    output_skewr2_values = 'r_rate'
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
