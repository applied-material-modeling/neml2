# Translated from tests/unit/models/common/ArrheniusParameter.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'p'
    input_Scalar_names = 'T'
    input_Scalar_values = '1000'
    output_Scalar_names = 'p'
    output_Scalar_values = 'p_correct'
  []
[]

[Models]
  [p]
    type = ArrheniusParameter
    temperature = 'T'
    reference_value = 'p0'
    activation_energy = 'Q'
    ideal_gas_constant = 8.314
  []
[]

[Tensors]
  [p0]
    type = Python
    expr = "Scalar(torch.linspace(1.0, 10.0, 5, dtype=torch.float64))"
  []
  [Q]
    type = Python
    expr = "Scalar(torch.linspace(1.0e3, 2.0e4, 5, dtype=torch.float64))"
  []
  [p_correct]
    type = Python
    expr = "Scalar(torch.tensor([0.8866729736328125, 1.6275087594985962, 1.5555329322814941, 1.2379260063171387, 0.9021309018135071], dtype=torch.float64))"
  []
[]
