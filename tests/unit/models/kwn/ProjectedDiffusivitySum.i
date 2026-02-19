[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/x1 state/x2'
    input_Scalar_values = 'x1 x2'
    output_Scalar_names = 'state/sum'
    output_Scalar_values = 'sum'
    check_AD_parameter_derivatives = false
  []
[]

[Tensors]
  [dx1]
    type = Scalar
    values = '0.2 0.1'
    batch_shape = '(2)'
  []
  [dx2]
    type = Scalar
    values = '0.05 0.15'
    batch_shape = '(2)'
  []
  [x1]
    type = Scalar
    values = '0.4 0.6'
    batch_shape = '(2)'
  []
  [x2]
    type = Scalar
    values = '0.8 0.5'
    batch_shape = '(2)'
  []
  [sum]
    type = Scalar
    values = '0.215625 0.25833333333333336'
    batch_shape = '(2)'
  []
[]

[Models]
  [model]
    type = ProjectedDiffusivitySum
    concentration_differences = 'dx1 dx2'
    diffusivities = '0.5 0.2'
    far_field_concentrations = 'state/x1 state/x2'
    projected_diffusivity_sum = 'state/sum'
  []
[]
