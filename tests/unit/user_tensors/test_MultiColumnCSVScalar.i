[Tensors]
  [a]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar.csv'
    batch_shape = '(3,2)'
  []
  [b]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar.csv'
    csv_columns = 'col1 col2'
    batch_shape = '(2,2)'
  []
[]