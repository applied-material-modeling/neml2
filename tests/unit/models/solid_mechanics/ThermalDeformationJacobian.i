[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'forces/T'
    input_Scalar_values = '400'
    output_R2_names = 'state/J'
    output_R2_values = 1.001
    parameter_derivative_rel_tol = 1e-04
  []
[]

[Models]
  [model]
    type = ThermalDeformationJacobian
    reference_temperature = 300
    CTE = 1e-5
    jacobian = 'state/J'
  []
[]
