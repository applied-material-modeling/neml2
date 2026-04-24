[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    output_Scalar_names = 'slip_hardening_rate'
    output_Scalar_values = 'rate'
    input_Scalar_names = 'slip_hardening sum_slip_rates'
    input_Scalar_values = 'tau_bar sum_slip'
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
    values = '20.0'
  []
[]

[Models]
  [model]
    type = LinearSingleSlipHardeningRule
    hardening_slope = 200.0
  []
[]
