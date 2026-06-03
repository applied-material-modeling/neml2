# Translated from tests/unit/models/solid_mechanics/traction_separation_law/OrthotropicLinearTraction.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'normal_separation tangential_separation_1 tangential_separation_2'
    input_Scalar_values = '0.02 0.005 -0.003'
    output_Vec_names = 'traction'
    output_Vec_values = 'T_expected'
    derivative_abs_tol = 1e-6
  []
[]

[Tensors]
  [T_expected]
    type = Python
    # T_n  = 1000 * 0.02   = 20
    # T_s1 = 200  * 0.005  = 1
    # T_s2 = 200  * -0.003 = -0.6
    expr = 'Vec(torch.tensor([20.0, 1.0, -0.6]))'
  []
[]

[Models]
  [model]
    type = OrthotropicLinearTraction
    normal_stiffness = 1000.0
    tangential_stiffness = 200.0
  []
[]
