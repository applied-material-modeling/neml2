# Translated from tests/unit/models/solid_mechanics/ScalarTwoStageThermalAnnealing.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'temperature base_rate base'
    input_Scalar_values = 'temperature_in 20.0 100.0'
    output_Scalar_names = 'modified_rate'
    output_Scalar_values = 'correct_values'
  []
[]

[Models]
  [model]
    type = ScalarTwoStageThermalAnnealing
    base_rate = 'base_rate'
    base = 'base'
    modified_rate = 'modified_rate'
    temperature = 'temperature'

    T1 = 1000.0
    T2 = 1200.0

    tau = 20.0
  []
[]

[Tensors]
  [temperature_in]
    type = Python
    expr = "Scalar(torch.tensor([800.0, 999.0, 1001.0, 1100.0, 1199.0, 1201.0, 1250.0]))"
  []

  [correct_values]
    type = Python
    expr = "Scalar(torch.tensor([20.0, 20.0, 0.0, 0.0, 0.0, -5.0, -5.0]))"
  []
[]
