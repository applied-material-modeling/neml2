# Unit test for PowerLawCreepFlowRate (temperature-independent form, Q = 0):
#   gamma_dot = A * <sigma_e>^n = 1e-30 * (1e8)^4 = 1e2
# This is the rate law of the MOOSE ADPowerLawCreepStressUpdate object in the
# metallic-fuel deck x447_dp11_fuel_moose.i (A = 1e-30, n = 4, Q = 0).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'yield_function'
    input_Scalar_values = 'sigma_e'
    output_Scalar_names = 'flow_rate'
    output_Scalar_values = 'rate_expected'
  []
[]

[Tensors]
  [sigma_e]
    type = Python
    expr = 'Scalar(1.0e8)'
  []
  [rate_expected]
    type = Python
    expr = 'Scalar(1.0e2)'
  []
[]

[Models]
  [model]
    type = PowerLawCreepFlowRate
    coefficient = 1.0e-30
    exponent = 4
  []
[]
