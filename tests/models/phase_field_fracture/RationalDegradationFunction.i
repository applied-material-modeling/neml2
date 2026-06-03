# Translated from tests/unit/models/phase_field_fracture/RationalDegradationFunction.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'degrade'
    input_Scalar_names = 'd'
    input_Scalar_values = '0.787'
    output_Scalar_names = 'g'
    output_Scalar_values = '0.0212478'
    derivative_abs_tol = 1e-06
  []
[]

[Tensors]
  [p]
    type = Python
    expr = 'Scalar(torch.tensor(2.0, dtype=torch.float64))'
  []
[]

[Models]
  [degrade]
    type = RationalDegradationFunction
    phase = 'd'
    degradation = 'g'
    power = 'p'
    b1 = 1
    b2 = 1.3868
    b3 = 0.6567
  []
[]
