[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'temperature base_rate base'
    input_Scalar_values = 'temperature_in 20.0 100.0'
    output_Scalar_names = 'modified_rate'
    output_Scalar_values = 'correct_values'
    check_second_derivatives = false
    check_AD_parameter_derivatives = false
  []
[]

[Models]
  [model]
    type = ScalarTwoStageThermalAnnealing
    base_rate = 'base_rate'
    base = 'base'
    modified_rate = 'modified_rate'
    temperature = 'temperature'

    T1 = 1000.0
    T2 = 1200.0

    tau = 20.0
  []
[]

[Tensors]
  [temperature_in]
    type = Scalar
    values = "800.0 999 1001 1100 1199 1201 1250"
    batch_shape = '(7)'
  []

  [correct_values]
    type = Scalar
    values = "20.0 20.0 0.0 0.0 0.0 -5.0 -5.0"
    batch_shape = '(7)'
  []
[]
