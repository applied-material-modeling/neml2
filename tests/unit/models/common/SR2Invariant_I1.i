[Tensors]
  [foo]
    type = FillSR2
    values = '1 2 3 4 5 6'
    # This tensor reads
    # A = [ 1 6 5
    #       6 2 4
    #       5 4 3 ]
    # I1(A) = 6
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'internal/O'
    input_SR2_values = 'foo'
    output_Scalar_names = 'internal/I1'
    output_Scalar_values = '6'
    check_second_derivatives = true
  []
[]

[Models]
  [model]
    type = SR2Invariant
    invariant_type = 'I1'
    tensor = 'internal/O'
    invariant = 'internal/I1'
  []
[]
