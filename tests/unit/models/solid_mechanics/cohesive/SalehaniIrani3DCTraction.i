[Tensors]
  [delta]
    type = Vec
    values = '0.5 0.1 0.05'
  []
  [T_expected]
    type = Vec
    values = '81.922444868181131 19.875389812467894 9.937694906233947'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Vec_names = 'forces/displacement_jump'
    input_Vec_values = 'delta'
    output_Vec_names = 'state/traction'
    output_Vec_values = 'T_expected'
    check_second_derivatives = false
  []
[]

[Models]
  [model]
    type = SalehaniIrani3DCTraction
    normal_gap_at_maximum_normal_traction = 1.0
    tangential_gap_at_maximum_shear_traction = 1.0
    maximum_normal_traction = 100
    maximum_shear_traction = 200
  []
[]
