# Translated from tests/unit/models/solid_mechanics/plasticity/KocksMeckingRateSensitivity.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'p'
    input_Scalar_names = 'temperature'
    input_Scalar_values = '1000'
    output_Scalar_names = 'rate_sensitivity'
    output_Scalar_values = 'p_correct'
  []
[]

[Models]
  [p]
    type = KocksMeckingRateSensitivity
    shear_modulus = 'mu'
    A = 'A'
    k = 1.38064e-20
    b = 2.019e-7
    temperature = 'temperature'
  []
[]

[Tensors]
  [mu]
    type = Python
    expr = "Scalar(torch.linspace(50000.0, 100000.0, 5, dtype=torch.float64))"
  []
  [A]
    type = Python
    expr = "Scalar(torch.linspace(-3.5, -5.5, 5, dtype=torch.float64))"
  []
  [p_correct]
    type = Python
    expr = "Scalar(torch.tensor([8.51589828, 9.31426374, 9.93521466, 10.43197539, 10.83841599], dtype=torch.float64))"
  []
[]
