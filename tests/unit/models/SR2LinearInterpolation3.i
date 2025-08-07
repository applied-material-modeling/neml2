[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'D'
    input_Scalar_names = 'forces/T'
    input_Scalar_values = '300'
    output_SR2_names = 'parameters/D'
    output_SR2_values = 'DT'
    check_second_derivatives = true
    check_AD_parameter_derivatives = false
  []
[]

[Models]
  [D]
    type = SR2LinearInterpolation
    argument = 'forces/T'
    abscissa = 'T'
    ordinate = 'D'
  []
[]

[Tensors]
  [T]
    type = LinspaceScalar
    start = 273.15
    end = 2000
    nstep = 100
    group = 'intermediate'
  []
  [d0]
    type = FullScalar
    batch_shape = '(5,1)'
    value = 1
  []
  [D0]
    type = FillSR2
    values = 'd0'
  []
  [d1]
    type = FullScalar
    batch_shape = '(5,1)'
    value = 30
  []
  [D1]
    type = FillSR2
    values = 'd1'
  []
  [D]
    type = LinspaceSR2
    start = 'D0'
    end = 'D1'
    nstep = 100
    group = 'intermediate'
  []
  [dT]
    type = FullScalar
    value = 1.4509077221530537
  []
  [DT]
    type = FillSR2
    values = 'dT'
  []
[]
