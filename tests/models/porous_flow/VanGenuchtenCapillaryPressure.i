# Translated from tests/unit/models/porous_flow/VanGenuchtenCapillaryPressure.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'S'
    input_Scalar_values = 'S'
    output_Scalar_names = 'Pc'
    output_Scalar_values = 'Pc'
    derivative_abs_tol = 1e-7
  []
[]

[Tensors]
  [S]
    type = Python
    expr = "Scalar(torch.tensor([0.05, 0.2, 0.65, 0.9], dtype=torch.float64))"
  []
  [Pc]
    type = Python
    expr = "Scalar(torch.tensor([1.104503654476773, 0.643666536951943, 0.3175165612, 0.1932327962], dtype=torch.float64))"
  []
[]

[Models]
  [model]
    type = VanGenuchtenCapillaryPressure
    a = 0.333333333333333
    m = 0.7
    effective_saturation = 'S'
    capillary_pressure = 'Pc'
    log_extension = true
    transition_saturation = 0.1
  []
[]
