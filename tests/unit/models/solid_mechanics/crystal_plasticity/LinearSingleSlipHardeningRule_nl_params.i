[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    output_Scalar_names = 'state/internal/slip_hardening_rate'
    output_Scalar_values = 'rate'
    input_Scalar_names = 'state/internal/slip_hardening state/internal/sum_slip_rates state/theta'
    input_Scalar_values = 'tau_bar sum_slip 200.0'
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
  [model0]
    type = LinearSingleSlipHardeningRule
    hardening_slope = 'state/theta'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
