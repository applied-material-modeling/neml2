[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/J'
    input_Scalar_values = 0.8
    input_R2_names = 'state/F state/P'
    input_R2_values = 'F P'
    output_Scalar_names = 'state/pc'
    output_Scalar_values = '46.83333333333333333'
    derivative_abs_tol = 1e-06
  []
[]

[Tensors]
  [F]
    type = FillR2
    values = "0.2 0.5 0.3
              1.0 8.0 7.0
              6.0 5.0 2.0"
  []
  [P]
    type = FillR2
    values = "3.0 2.0 4.0
              9.8 1.2 3.3
              4.4 7.3 2.1"
  []
[]

[Models]
  [model]
    type = AdvectionStress
    coefficient = 1.0
    jacobian = 'state/J'
    deformation_gradient = 'state/F'
    pk1_stress = 'state/P'
    average_advection_stress = 'state/pc'
  []
[]
