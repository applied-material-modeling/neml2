[Tensors]
  [scalar1]
    type = Scalar
    values = '1 2 3 4 5'
    batch_shape = '(5)'
  []
  [scalar2]
    type = Scalar
    values = '5 6 7 8 9'
    batch_shape = '(5)'
  []
  [scalar3]
    type = Scalar
    values = '-1 -2 -3 -4 -5'
    batch_shape = '(5)'
  []
  [auto_3_crossref]
    type = FillSR2
    values = 'scalar1 scalar2 scalar3'
  []
[]
