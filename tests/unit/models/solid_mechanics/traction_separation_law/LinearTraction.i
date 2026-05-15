[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Vec_names = 'displacement_jump'
    input_Vec_values = 'jump'
    output_Vec_names = 'traction'
    output_Vec_values = 'traction'
    derivative_abs_tol = 1e-6
  []
[]

[Tensors]
  [jump]
    type = Vec
    values = '0.02 0.005 -0.003'
  []
  # T_n  = 1000 * 0.02   = 20
  # T_s1 = 200  * 0.005  = 1
  # T_s2 = 200  * -0.003 = -0.6
  [traction]
    type = Vec
    values = '20 1 -0.6'
  []
[]

[Models]
  [model]
    type = LinearTraction
    normal_stiffness = 1000
    tangential_stiffness = 200
  []
[]
