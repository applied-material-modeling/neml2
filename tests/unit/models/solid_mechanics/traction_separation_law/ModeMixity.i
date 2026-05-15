[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'dn_pos ds'
    input_Scalar_values = '0.5 0.5'
    output_Scalar_names = 'beta'
    # beta = ds / dn_pos = 0.5 / 0.5 = 1.0  (opening branch)
    output_Scalar_values = '1.0'
    derivative_abs_tol = 1e-6
  []
[]

[Models]
  [model]
    type = ModeMixity
    normal = 'dn_pos'
    tangential = 'ds'
    to = 'beta'
  []
[]
