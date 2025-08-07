[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_Scalar_names = 'forces/T'
    input_Scalar_values = '300'
    output_Scalar_names = 'parameters/E'
    output_Scalar_values = '188911.6020499754'
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
    group = 'intermediate'
  []
  [E]
    type = LinspaceScalar
    start = 1.9e5
    end = 1.2e5
    nstep = 100
    dim = 0
    group = 'intermediate'
  []
[]
