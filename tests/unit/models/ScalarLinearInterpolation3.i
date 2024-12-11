[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_scalar_names = 'forces/T'
    input_scalar_values = '300'
    output_scalar_names = 'parameters/E'
    output_scalar_values = '188911.6020499754'
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
    start = 273.15
    end = 2000
    nstep = 100
    dim = 0
  []
  [E0]
    type = FullScalar
    batch_shape = '(5,1)'
    value = 1.9e5
  []
  [E1]
    type = FullScalar
    batch_shape = '(5,1)'
    value = 1.2e5
  []
  [E]
    type = LinspaceScalar
    start = 'E0'
    end = 'E1'
    nstep = 100
    dim = 2
  []
[]
