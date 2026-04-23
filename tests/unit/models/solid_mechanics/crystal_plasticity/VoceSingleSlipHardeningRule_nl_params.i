[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    output_Scalar_names = 'slip_hardening_rate'
    output_Scalar_values = 'rate'
    input_Scalar_names = 'slip_hardening sum_slip_rates theta0 tau_f'
    input_Scalar_values = 'tau_bar sum_slip 200.0 60.0'
  []
[]

[Tensors]
  [tau_bar]
    type = Scalar
    values = '40.0'
  []
  [sum_slip]
    type = Scalar
    values = '0.1'
  []
  [rate]
    type = Scalar
    values = '6.666666666666667'
  []
[]

[Models]
  [model0]
    type = VoceSingleSlipHardeningRule
    initial_slope = 'theta0'
    saturated_hardening = 'tau_f'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
