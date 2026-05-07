[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/mag'
    input_Scalar_values = 'mag'
    output_Scalar_names = 'state/source'
    output_Scalar_values = 'source'
    output_with_intrsc_intmd_dims = 'state/source'
    output_intrsc_intmd_dims = '1'
    check_AD_parameter_derivatives = false
  []
[]

[Tensors]
  [mag]
    type = Scalar
    values = '2.5'
  []
  [centers]
    type = Scalar
    values = '0.1 0.5 1.0'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [source]
    type = Scalar
    values = '2.5 0.0 0.0'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
[]

[Models]
  [model]
    type = DumpInSmallestBin
    magnitude = 'state/mag'
    cell_centers = 'centers'
    dumped_source = 'state/source'
  []
[]
