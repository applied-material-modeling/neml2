# Translated from tests/unit/models/solid_mechanics/crystal_plasticity/OrientationRate.i
# (FillSR2 6-value -> SR2.fill; FillWR2 -> WR2(torch.tensor(...))).
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
    type = Python
    expr = 'SR2.fill(0.100, 0.110, 0.100, 0.050, 0.040, 0.030)'
  []
  [dp]
    type = Python
    expr = 'SR2.fill(0.050, 0.120, 0.080, 0.010, 0.010, 0.090)'
  []
  [w]
    type = Python
    expr = 'WR2(torch.tensor([0.01, -0.02, 0.03]))'
  []
  [wp]
    type = Python
    expr = 'WR2(torch.tensor([0.03, 0.02, -0.01]))'
  []
  [r_rate]
    type = Python
    expr = 'WR2(torch.tensor([-0.0252, -0.037, 0.0411]))'
  []
[]

[Models]
  [model]
    type = OrientationRate
  []
[]
