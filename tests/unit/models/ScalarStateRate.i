[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    batch_shape = '(10)'
    input_scalar_names = 'state/foo old_state/foo forces/t old_forces/t'
    input_scalar_values = '-0.3 0 1.3 1.1'
    output_scalar_names = 'state/foo_rate'
    output_scalar_values = '-1.5'

    check_values = true
    check_first_derivatives = true
    check_second_derivatives = false
    check_AD_first_derivatives = false
    check_AD_second_derivatives = false
    check_AD_derivatives = false
    check_AD_parameter_derivatives = false
    check_cuda = false
  []
[]

[Models]
  [model]
    type = ScalarVariableRate
    variable = 'state/foo'
    rate = 'state/foo_rate'
  []
[]
