[Tensors]
  [T_vals]
    type = Scalar
    values = '0 100 200'
    batch_shape = '(3)'
  []
  [c_A_vals]
    type = Scalar
    values = '1 2 3'
    batch_shape = '(3)'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'forces/T state/A state/B'
    input_Scalar_values = '100 0.5 1.5'
    output_Scalar_names = 'state/C'
    output_Scalar_values = '9.25'
    check_AD_parameter_derivatives = false
  []
[]

[Models]
  [model]
    type = ArchivedModel
    archive = 'ComposedModel5_model.gz'
  []
[]
