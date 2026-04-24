[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'foo bar temperature'
    input_Scalar_values = '1 2 100'
    input_SR2_names = 'baz'
    input_SR2_values = '0.5'
    output_Scalar_names = 'foo_rate bar_rate'
    output_Scalar_values = '301.5 -89.02'
    output_SR2_names = 'baz_rate'
    output_SR2_values = '145.5'
    check_values = true
    check_derivatives = true
    check_second_derivatives = false
  []
[]

[Models]
  [model]
    type = ADSampleRateModel
  []
[]
