[Tensors]
  [batch_foo]
    type = Tensor
    base_shape = '(5)'
    values = '-0.3 1.1 0 1.3 1.1'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Tensor_names = 'state/batched_foo'
    input_Tensor_values = 'batch_foo'
    input_Scalar_names = 'state/foo'
    input_Scalar_values = '-0.3'
    output_Scalar_names = 'residual/foo'
    output_Scalar_values = '-0.94'
  []
[]

[Models]
  [model]
    type = ScalarImplicitListReduction
    variable = 'state/foo'
    batched_variable = 'state/batched_foo'
    batched_variable_list_size = 5
  []
[]
