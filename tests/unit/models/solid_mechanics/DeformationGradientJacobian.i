[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_R2_names = 'state/F'
    input_R2_values = 'F'
    output_Scalar_names = 'state/J'
    output_Scalar_values = '3.3'
    derivative_abs_tol = 1e-06
  []
[]

[Tensors]
  [F]
    type = FillR2
    values = '0.2 0.5 0.3 1.0 8.0 7.0 6.0 5.0 2.0'
  []
[]

[Models]
  [model]
    type = DeformationGradientJacobian
    deformation_gradient = 'state/F'
    jacobian = 'state/J'
  []
[]
