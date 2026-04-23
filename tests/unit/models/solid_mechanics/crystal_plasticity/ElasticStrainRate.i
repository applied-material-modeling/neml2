[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'elastic_strain deformation_rate plastic_deformation_rate'
    input_SR2_values = 'e d dp'
    input_WR2_names = 'vorticity'
    input_WR2_values = 'w'
    output_SR2_names = 'elastic_strain_rate'
    output_SR2_values = 'e_rate'
  []
[]

[Tensors]
  [e]
    type = FillSR2
    values = '0.100 0.110 0.100 0.050 0.040 0.030'
  []
  [d]
    type = FillSR2
    values = '0.050 -0.010 0.020 0.040 0.030 -0.060'
  []
  [dp]
    type = FillSR2
    values = '0.050 0.120 0.080 0.010 0.010 0.090'
  []
  [w]
    type = FillWR2
    values = '0.01 -0.02 0.03'
  []
  [e_rate]
    type = FillSR2
    values = '-0.0034 -0.1292 -0.0574 0.0319 0.0188 -0.1517'
  []
[]

[Models]
  [model]
    type = ElasticStrainRate
  []
[]
