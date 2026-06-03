# Translated from tests/unit/models/solid_mechanics/elasticity/GreenLagrangeStrain.i
# (FillR2 9-val -> R2 row-major; FillSR2 6-val -> SR2.fill with sqrt(2) Mandel scaling).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_R2_names = 'deformation_gradient'
    input_R2_values = 'F'
    output_SR2_names = 'strain'
    output_SR2_values = 'E'
    derivative_abs_tol = 1e-5
    derivative_rel_tol = 0
  []
[]

[Tensors]
  [F]
    type = Python
    expr = 'R2(torch.tensor([[1.01, 0.01, 0.02], [0.002, 0.93, 0.009], [0.05, -0.02, 1.1]]))'
  []
  [E]
    type = Python
    expr = 'SR2.fill(0.011302, -0.0673, 0.1052405, -0.006715, 0.037609, 0.00548)'
  []
[]

[Models]
  [model]
    type = GreenLagrangeStrain
    deformation_gradient = 'deformation_gradient'
    strain = 'strain'
  []
[]
