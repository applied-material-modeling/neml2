# Native unit test for SR2Determinant.
# SR2.fill(2, 3, 5) -> pure diagonal symmetric tensor; det = 2*3*5 = 30.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'F'
    input_SR2_values = 'F'
    output_Scalar_names = 'J'
    output_Scalar_values = '30.0'
    derivative_abs_tol = 1e-06
  []
[]

[Tensors]
  [F]
    type = Python
    expr = 'SR2.fill(2.0, 3.0, 5.0)'
  []
[]

[Models]
  [model]
    type = SR2Determinant
    input = 'F'
    determinant = 'J'
  []
[]
