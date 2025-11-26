[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/slip_hardening state/const'
    input_Scalar_values = 'hardening 50.0'
    output_Scalar_names = 'state/internal/slip_strengths'
    output_Scalar_values = '150'
  []
[]

[Tensors]
  [hardening]
    type = Scalar
    values = '100.0'
  []
[]

[Models]
  [model0]
    type = SingleSlipStrengthMap
    constant_strength = 'state/const'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
