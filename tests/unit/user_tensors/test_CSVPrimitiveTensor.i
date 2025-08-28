[Tensors]
  [a]
    type = CSVScalar
    csv_file = 'user_tensors/CSVPrimitiveTensor.csv'
    tensor_name = 'scalar'
    batch_shape = '(4)'
  []
  [b]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor.csv'
    tensor_name = 'disp'
    batch_shape = '(4)'
  []
  [c]
    type = CSVSR2
    csv_file = 'user_tensors/CSVPrimitiveTensor.csv'
    tensor_name = 's'
    batch_shape = '(4)'
  []
[]