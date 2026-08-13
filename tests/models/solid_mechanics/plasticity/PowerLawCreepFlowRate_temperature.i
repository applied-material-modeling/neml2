# Unit test for PowerLawCreepFlowRate with the (optional) Arrhenius term active:
#   gamma_dot = A * <sigma_e>^n * exp(-Q/(R T))
#             = 1e-30 * (1e8)^4 * exp(-1.5e5 / (8.3143 * 900))
# Exercises the temperature input path and its d/dT JVP.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'yield_function temperature'
    input_Scalar_values = 'sigma_e temp'
    output_Scalar_names = 'flow_rate'
    output_Scalar_values = 'rate_expected'
  []
[]

[Tensors]
  [sigma_e]
    type = Python
    expr = 'Scalar(1.0e8)'
  []
  [temp]
    type = Python
    expr = 'Scalar(900.0)'
  []
  [rate_expected]
    type = Python
    expr = 'Scalar((1.0e-30 * (1.0e8)**4) * torch.exp(torch.tensor(-1.5e5 / (8.3143 * 900.0), dtype=torch.float64)))'
  []
[]

[Models]
  [model]
    type = PowerLawCreepFlowRate
    coefficient = 1.0e-30
    exponent = 4
    activation_energy = 1.5e5
    gas_constant = 8.3143
    temperature = 'temperature'
  []
[]
