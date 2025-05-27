[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'forces/T'
    input_Scalar_values = '400'
    output_R2_names = 'state/F'
    output_R2_values = 'Ft'
    parameter_derivative_rel_tol = 1e-04
  []
[]

[Tensors]
  [Ft]
    type = FillR2
    values = '1.000333222222222 0.0   0.0
              0.0   1.000333222222222 0.0
              0.0   0.0   1.000333222222222'
  []
[]

[Models]
  [model]
    type = ThermalDeformationGradient
    reference_temperature = 300
    CTE = 1e-5
    deformation_gradient = 'state/F'
  []
[]
