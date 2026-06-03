# Translated from tests/unit/models/phase_field_fracture/PowerDegradationFunction.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'degrade'
    input_Scalar_names = 'd'
    input_Scalar_values = '0.787'
    output_Scalar_names = 'g'
    output_Scalar_values = '0.000438428'
    derivative_abs_tol = 1e-06
  []
[]

[Tensors]
  [p]
    type = Python
    expr = 'Scalar(torch.tensor(5.0, dtype=torch.float64))'
  []
[]

[Models]
  [degrade]
    type = PowerDegradationFunction
    phase = 'd'
    degradation = 'g'
    power = 'p'
  []
[]
