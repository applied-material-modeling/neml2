[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/S'
    input_Scalar_values = 'S'
    output_Scalar_names = 'state/Pc'
    output_Scalar_values = 'Pc'
    check_AD_parameter_derivatives = false
    check_second_derivatives = true
  []
[]

[Tensors]
  [S]
    type = Scalar
    values = '0.05 0.2 0.65 0.9'
    batch_shape = '(4)'
  []
  [Pc]
    type = Scalar
    values = '126.0403524130621 22.92220613 4.255929933 2.673595365'
    batch_shape = '(4)'
  []
[]

[Models]
  [model]
    type = BrooksCoreyPressure
    threshold_pressure = 2.3
    exponent = 0.7
    effective_saturation = 'state/S'
    capillary_pressure = 'state/Pc'
    log_extension = true
    transition_saturation = 0.1
  []
[]
