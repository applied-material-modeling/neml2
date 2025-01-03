[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/foo state/bar forces/temperature'
    input_Scalar_values = '1 2 100'
    input_SR2_names = 'state/baz'
    input_SR2_values = '0.5'
    output_Scalar_names = 'state/foo_rate state/bar_rate'
    output_Scalar_values = '301.5 -89.02'
    output_SR2_names = 'state/baz_rate'
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
