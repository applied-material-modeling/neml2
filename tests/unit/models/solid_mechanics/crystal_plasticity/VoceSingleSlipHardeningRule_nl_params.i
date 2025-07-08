[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    output_Scalar_names = 'state/internal/slip_hardening_rate'
    output_Scalar_values = 'rate'
    input_Scalar_names = 'state/internal/slip_hardening state/internal/sum_slip_rates state/theta0 state/tau_f'
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
    initial_slope = 'state/theta0'
    saturated_hardening = 'state/tau_f'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
