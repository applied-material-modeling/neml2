[Tensors]
  [batch_foo]
    type = Tensor
    base_shape = '(2,6)'
    values = '-0.3 1.1 0 1.3 1.1 0.5 
              0.8 0.2 1.05 0.8 0.2 1.05'
  []
  [foo]
    type = FullSR2
    batch_shape = '(2,)'
    value = 1.3
  []
  [rfoo]
    type = SR2
    batch_shape = '(2,)'
    values = '1.05 0.65 0.775 0.25 0.65 0.525'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Tensor_names = 'state/batched_foo'
    input_Tensor_values = 'batch_foo'
    input_SR2_names = 'state/foo'
    input_SR2_values = 'foo'
    output_SR2_names = 'residual/foo'
    output_SR2_values = 'rfoo'
  []
[]

[Models]
  [model]
    type = SR2ImplicitListReduction
    variable = 'state/foo'
    batched_variable = 'state/batched_foo'
    batched_variable_list_size = 2
  []
[]
