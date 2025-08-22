[Tensors]
  [a]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/test_MultiColumnCSVScalar/MultiColumnCSVScalar.csv'
    column_names = 'col1 col2 col3'
    batch_shape = '(3,2)'
  []
  [b]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/test_MultiColumnCSVScalar/MultiColumnCSVScalarSpace.csv'
    column_indices = '0 1 2'
    delimiter = 'SPACE'
    batch_shape = '(3,2)'
  []
  [c]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/test_MultiColumnCSVScalar/MultiColumnCSVScalarSemicolon.csv'
    column_names = 'col1 col2'
    delimiter = 'SEMICOLON'
    batch_shape = '(2,2)'
  []
  [d]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/test_MultiColumnCSVScalar/MultiColumnCSVScalarTab.txt'
    column_indices = '0 1'
    delimiter = 'TAB'
    batch_shape = '(2,2)'
  []
  [e]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/test_MultiColumnCSVScalar/MultiColumnCSVScalarNoHeader.csv'
    column_indices = '0 1 2'
    header_row = false
    batch_shape = '(3,2)'
  []
  [f]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/test_MultiColumnCSVScalar/MultiColumnCSVScalarNoHeaderSpace.csv'
    column_indices = '0 1'
    header_row = false
    delimiter = 'SPACE'
    batch_shape = '(2,2)'
  []
  [g]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/test_MultiColumnCSVScalar/MultiColumnCSVScalar.csv'
    column_names = 'col1 col2 col3'
    column_indices = '0 1 2'
    batch_shape = '(3,2)'
  []
  [h]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/test_MultiColumnCSVScalar/MultiColumnCSVScalarNoHeader.csv'
    column_names = 'col1 col2 col3'
    header_row = false
    batch_shape = '(3,2)'
  []
  [i]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/test_MultiColumnCSVScalar/MultiColumnCSVScalar.csv'
    batch_shape = '(3,2)'
  []
[]