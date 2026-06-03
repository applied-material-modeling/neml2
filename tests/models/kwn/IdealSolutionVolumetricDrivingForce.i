# Translated from tests/unit/models/kwn/IdealSolutionVolumetricDrivingForce.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'forces/T state/x1 state/x2'
    input_Scalar_values = 'T x1 x2'
    output_Scalar_names = 'state/dg_v'
    output_Scalar_values = 'dg_v'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = "Scalar(torch.tensor(300.0, dtype=torch.float64))"
  []
  [x1]
    type = Python
    expr = "Scalar(torch.tensor(0.2, dtype=torch.float64))"
  []
  [x2]
    type = Python
    expr = "Scalar(torch.tensor(0.5, dtype=torch.float64))"
  []
  # dg_v = R T * [ln(0.2/0.1) + ln(0.5/0.1)]
  #      = 8.314 * 300 * ln(10)
  #      = 2494.2 * 2.302585092994046
  [dg_v]
    type = Python
    expr = "Scalar(torch.tensor(5743.107738945748, dtype=torch.float64))"
  []
[]

[Models]
  [model]
    type = IdealSolutionVolumetricDrivingForce
    temperature = 'forces/T'
    current_concentrations = 'state/x1 state/x2'
    equilibrium_concentrations = '0.1 0.1'
    gas_constant = 8.314
    driving_force = 'state/dg_v'
  []
[]
