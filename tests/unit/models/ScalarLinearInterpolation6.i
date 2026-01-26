[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_Scalar_names = 'forces/T'
    input_Scalar_values = 'temp'
    output_Scalar_names = 'parameters/E'
    output_Scalar_values = 'interp'
    check_second_derivatives = true
    check_AD_parameter_derivatives = false
  []
[]

[Models]
  [E]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = 'T'
    ordinate = 'E' 
  []
[]

[Tensors]
  [T]
    type = LinspaceScalar
    start = 600
    end = 1350
    nstep = 2
    dim = 0
  []
  [E]
    type = LinspaceScalar
    start = -1.24080
    end = -1.42840
    nstep = 2
    dim = 0
  []
  [temp]
    type = Scalar
    values = '1350 1350'
    batch_shape = '(2)'
  []
  [interp]
    type = Scalar
    values = '-1.4284 -1.4284'
    batch_shape = '(2)'
  []
[]
