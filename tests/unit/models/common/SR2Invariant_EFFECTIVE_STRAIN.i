[Tensors]
  [foo]
    type = FillSR2
    values = '1 2 3 4 5 6'
    # This tensor reads
    # A = [ 1 6 5
    #       6 2 4
    #       5 4 3 ]
    # EFFECTIVE_STRAIN(A) = 10.583005
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'internal/O'
    input_SR2_values = 'foo'
    output_Scalar_names = 'internal/I2'
    output_Scalar_values = '10.5830052'
    check_second_derivatives = true
  []
[]

[Models]
  [model]
    type = SR2Invariant
    invariant_type = 'EFFECTIVE_STRAIN'
    tensor = 'internal/O'
    invariant = 'internal/I2'
  []
[]
