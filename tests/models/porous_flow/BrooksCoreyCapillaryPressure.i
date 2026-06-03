# Translated from tests/unit/models/porous_flow/BrooksCoreyCapillaryPressure.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'S'
    input_Scalar_values = 'S'
    output_Scalar_names = 'Pc'
    output_Scalar_values = 'Pc'
  []
[]

[Tensors]
  [S]
    type = Python
    expr = "Scalar(torch.tensor([0.05, 0.2, 0.65, 0.9], dtype=torch.float64))"
  []
  [Pc]
    type = Python
    expr = "Scalar(torch.tensor([126.0403524130621, 22.92220613, 4.255929933, 2.673595365], dtype=torch.float64))"
  []
[]

[Models]
  [model]
    type = BrooksCoreyCapillaryPressure
    threshold_pressure = 2.3
    exponent = 0.7
    effective_saturation = 'S'
    capillary_pressure = 'Pc'
    log_extension = true
    transition_saturation = 0.1
  []
[]
