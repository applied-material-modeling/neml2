# GeneralElasticity with elastic_stiffness_tensor promoted to a runtime input
# (wired to a variable name rather than a [Tensors] literal), so the
# ModelUnitTest driver checks the promoted-parameter chain-rule action against
# autograd.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'strain'
    input_SR2_values = 'Ee'
    input_MRP_names = 'orientation'
    input_MRP_values = 'R'
    input_SSR4_names = 'elastic_tensor'
    input_SSR4_values = 'C'
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
    elastic_stiffness_tensor = 'elastic_tensor'
  []
[]
