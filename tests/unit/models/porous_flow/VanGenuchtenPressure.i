[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/S'
    input_Scalar_values = 'S'
    output_Scalar_names = 'state/Pc'
    output_Scalar_values = 'Pc'
    check_AD_parameter_derivatives = false
    derivative_abs_tol = 1e-7
    check_second_derivatives = true
    second_derivative_abs_tol = 1e-6
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
    values = '1.104503654476773 0.643666536951943 0.3175165612 0.1932327962'
    batch_shape = '(4)'
  []
[]

[Models]
  [model]
    type = VanGenuchtenPressure
    a = 0.333333333333333
    m = 0.7
    effective_saturation = 'state/S'
    capillary_pressure = 'state/Pc'
    log_extension = true
    transition_saturation = 0.1
  []
[]
