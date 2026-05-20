# neml2
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Vec_names = 'jump'
    input_Vec_values = 'jump_value'
    output_Scalar_names = 'sn ss1 ss2'
    output_Scalar_values = '0.4 0.5 0.6'
    derivative_abs_tol = 1e-6
  []
[]

[Tensors]
  [jump_value]
    type = Vec
    values = '0.4 0.5 0.6'
  []
[]

[Models]
  [model]
    type = VecComponents
    from = 'jump'
    to = 'sn ss1 ss2'
  []
[]
