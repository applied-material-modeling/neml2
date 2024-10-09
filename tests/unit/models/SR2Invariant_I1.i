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
    input_symr2_names = 'state/internal/O'
    input_symr2_values = 'foo'
    output_scalar_names = 'state/internal/I1'
    output_scalar_values = '6'
    check_second_derivatives = true
  []
[]

[Models]
  [model]
    type = SR2Invariant
    invariant_type = 'I1'
    tensor = 'state/internal/O'
    invariant = 'state/internal/I1'
  []
[]
