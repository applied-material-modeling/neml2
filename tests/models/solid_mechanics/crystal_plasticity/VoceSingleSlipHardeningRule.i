# Translated from tests/unit/models/.../VoceSingleSlipHardeningRule.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'slip_hardening sum_slip_rates'
    input_Scalar_values = '40.0 0.1'
    output_Scalar_names = 'slip_hardening_rate'
    output_Scalar_values = '6.666666666666667'
  []
[]

[Models]
  [model]
    type = VoceSingleSlipHardeningRule
    initial_slope = 200.0
    saturated_hardening = 60.0
  []
[]
