# Translated from tests/unit/models/common/VecComponents.i.
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
    type = Python
    expr = 'Vec(torch.tensor([0.4, 0.5, 0.6], dtype=torch.float64))'
  []
[]

[Models]
  [model]
    type = VecComponents
    from = 'jump'
    to = 'sn ss1 ss2'
  []
[]
