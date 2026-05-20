[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/mag state/loc'
    input_Scalar_values = 'mag loc'
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
    values = '2.0'
  []
  [loc]
    type = Scalar
    values = '0.2'
  []
  [centers]
    type = Scalar
    values = '-1.0 0.0 1.0'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [source]
    type = Scalar
    values = '0.3883721099664259 0.7820853879509118 0.5793831055229656'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
[]

[Models]
  [model]
    type = SmearedDeltaSource
    magnitude = 'state/mag'
    location = 'state/loc'
    width = 1.0
    cell_centers = 'centers'
    smeared_source = 'state/source'
  []
[]
