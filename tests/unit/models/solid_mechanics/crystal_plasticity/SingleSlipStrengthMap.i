[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/slip_hardening'
    input_Scalar_values = 'hardening'
    output_Tensor_names = 'state/internal/slip_strengths'
    output_Tensor_values = '150'
    output_with_intrsc_intmd_dims = 'state/internal/slip_strengths'
    output_intrsc_intmd_dims = '1'
    check_AD_parameter_derivatives = false
  []
[]

[Tensors]
  [hardening]
    type = Scalar
    values = '100.0'
    shape_manipulations = 'intmd_expand'
    shape_manipulation_args = '(2,3)'
  []
[]

[Models]
  [model]
    type = SingleSlipStrengthMap
    constant_strength = 50.0
  []
[]
