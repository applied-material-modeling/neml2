[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'temperature'
    input_Scalar_values = 'temperature_in'
    input_SR2_names = 'base_rate base'
    input_SR2_values = 'k_rate_in k_values_in'
    output_SR2_names = 'modified_rate'
    output_SR2_values = 'correct_values'
    check_second_derivatives = false
    check_AD_parameter_derivatives = false
    value_abs_tol = 1e-4
  []
[]

[Models]
  [model]
    type = SR2TwoStageThermalAnnealing
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
    type = SR2
    values = '-10.0000 15.0000 5.0000 -9.8995 21.2132 28.2843 -10.0000 15.0000 5.0000 -9.8995 21.2132 28.2843 -0.0000 0.0000 0.0000 -0.0000 0.0000 0.0000 -0.0000 0.0000 0.0000 -0.0000 0.0000 0.0000 -0.0000 0.0000 0.0000 -0.0000 0.0000 0.0000 -0.2500 -0.3500 -0.4000 -0.6364 -0.7071 -0.7778 -0.2500 -0.3500 -0.4000 -0.6364 -0.7071 -0.7778'
    batch_shape = '(7)'
  []

  [k_values_in]
    type = FillSR2
    values = '5.0 7.0 8.0 9.0 10.0 11.0'
  []

  [k_rate_in]
    type = FillSR2
    values = '-10 15 5 -7 15 20'
  []
[]
