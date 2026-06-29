# Translated from tests/unit/models/solid_mechanics/elasticity/GeneralElasticity.i:
# FillSR2 -> SR2.fill (6 values scale the shear by sqrt(2) into Mandel storage);
# FillRot -> Rodrigues vector; the SSR4 stiffness grid -> a (6, 6) Mandel matrix.
# The [Tensors] block forks to native type = Python expressions.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'strain'
    input_SR2_values = 'Ee'
    input_MRP_names = 'orientation'
    input_MRP_values = 'R'
    output_SR2_names = 'stress'
    output_SR2_values = 'S'
    derivative_rel_tol = 1e-4
    derivative_abs_tol = 1e-6
  []
[]

[Tensors]
  [Ee]
    type = Python
    expr = 'SR2.fill(0.09, 0.04, -0.02)'
  []
  [R]
    type = Python
    expr = 'MRP(torch.tensor([0.13991834, 0.18234513, 0.85043991]))'
  []
  [S]
    type = Python
    expr = 'SR2.fill(10.14791738, 4.65043712, -2.33254551, -0.74329637, 1.01251401, 1.25050411)'
  []
  [C]
    type = Python
    expr = '''
      SSR4(torch.tensor([
          [100,   2,   3,   4,   5,   6],
          [  7, 150,   9,  10,  11,  12],
          [ 13,  14, 300,  16,  17,  18],
          [ 19,  20,  21, 150,  23,  24],
          [ 25,  26,  27,  28, 200,  30],
          [ 31,  32,  33,  34,  35, 100],
      ], dtype=torch.float64))
    '''
  []
[]

[Models]
  [model]
    type = GeneralElasticity
    elastic_stiffness_tensor = 'C'
  []
[]
