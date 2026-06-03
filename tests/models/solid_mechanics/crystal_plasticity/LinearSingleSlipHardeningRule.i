# Translated from tests/unit/models/solid_mechanics/crystal_plasticity/LinearSingleSlipHardeningRule.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'slip_hardening sum_slip_rates'
    input_Scalar_values = '40.0 0.1'
    output_Scalar_names = 'slip_hardening_rate'
    output_Scalar_values = '20.0'
  []
[]

[Models]
  [model]
    type = LinearSingleSlipHardeningRule
    hardening_slope = 200.0
  []
[]
