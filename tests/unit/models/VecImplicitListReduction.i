[Tensors]
  [batch_foo]
    type = Tensor
    base_shape = '(2,3)'
    values = '-0.3 1.1 0 1.3 1.1 0.5'
  []
  [foo]
    type = FullVec
    batch_shape = '(2,)'
    value = 1.3
  []
  [rfoo]
    type = Vec
    batch_shape = '(2,)'
    values = '0.8 0.2 1.05 0.8 0.2 1.05'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Tensor_names = 'state/batched_foo'
    input_Tensor_values = 'batch_foo'
    input_Vec_names = 'state/foo'
    input_Vec_values = 'foo'
    output_Vec_names = 'residual/foo'
    output_Vec_values = 'rfoo'
  []
[]

[Models]
  [model]
    type = VecImplicitListReduction
    variable = 'state/foo'
    batched_variable = 'state/batched_foo'
    batched_variable_list_size = 2
  []
[]
