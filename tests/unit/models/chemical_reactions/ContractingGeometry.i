[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/alpha'
    input_Scalar_values = 'alpha'
    output_Scalar_names = 'state/out'
    output_Scalar_values = 'out'
    check_AD_parameter_derivatives = false
  []
[]

[Tensors]
  [alpha]
    type = Scalar
    values = '0.03 0.75 0.9'
    batch_shape = '(3)'
  []
  [out]
    type = Scalar
    values = '0.6437545 0.01546796 1.244795587e-3'
    batch_shape = '(3)'
  []
[]

[Models]
  [model]
    type = ContractingGeometry
    reaction_coef = 0.7
    reaction_order = 2.75
    conversion_degree = 'state/alpha'
    reaction_rate = 'state/out'
  []
[]
