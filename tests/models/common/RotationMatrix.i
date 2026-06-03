# MRP Rot → 3x3 rotation matrix R(r). Reference values computed from
# euler_rodrigues at r = (0.13991834, 0.18234513, 0.85043991) — the same
# orientation used in GeneralElasticity.i — to lock down the closed-form
# jvp_euler_rodrigues pushforward (no (..., 3, 3, 3) derivative kernel).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Rot_names = 'orientation'
    input_Rot_values = 'r'
    output_R2_names = 'orientation_matrix'
    output_R2_values = 'R'
  []
[]

[Tensors]
  [r]
    type = Python
    expr = 'Rot(torch.tensor([0.13991834, 0.18234513, 0.85043991], dtype=torch.float64))'
  []
  [R]
    type = Python
    expr = '''
      R2(torch.tensor([
          [-0.9185586555, -0.1767766912,  0.3535533875],
          [ 0.3061862117, -0.8838834796,  0.3535533881],
          [ 0.2500000002,  0.4330126971,  0.8660254061],
      ], dtype=torch.float64))
    '''
  []
[]

[Models]
  [model]
    type = RotationMatrix
    from = 'orientation'
    to = 'orientation_matrix'
  []
[]
