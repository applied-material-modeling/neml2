# Translated from tests/unit/models/common/ScalarLinearInterpolation1.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_Scalar_names = 'T'
    input_Scalar_values = '300'
    output_Scalar_names = 'E'
    output_Scalar_values = '188911.6020499754'
  []
[]

[Models]
  [E]
    type = ScalarLinearInterpolation
    argument = 'T'
    abscissa = 'T'
    ordinate = 'E'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = "Scalar(torch.linspace(273.15, 2000, 100, dtype=torch.float64)).sub_batch.retag(1)"
  []
  [E]
    type = Python
    expr = "Scalar(torch.linspace(1.9e5, 1.2e5, 100, dtype=torch.float64)).sub_batch.retag(1)"
  []
[]
