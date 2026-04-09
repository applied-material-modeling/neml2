[Tensors]
  [delta]
    type = Vec
    values = '0.01 0.02 0.005'
  []
  [T_expected]
    type = Vec
    values = '1.0 1.0 0.25'
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
    type = PureElasticTractionSeparation
    normal_stiffness = 100
    tangent_stiffness = 50
  []
[]
