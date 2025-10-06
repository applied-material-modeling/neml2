[Tensors]
  [all_columns]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data.csv'
  []
  [column_names]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data.csv'
    column_names = 'col3 col1'
  []
  [column_indices]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data.csv'
    column_indices = '2 0'
  []
  [delimiter]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data_semicolon.csv'
    delimiter = 'SEMICOLON'
  []
  [header_row]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data_comment.csv'
    header_row = 2
  []
  [no_header]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data_no_header.csv'
    no_header = true
  []
  [header_row_no_header]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data_no_header.csv'
    header_row = 3
    no_header = true
  []
  [row_major]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data.csv'
    indexing = 'ROW_MAJOR'
  []
  [batch_shape]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data.csv'
    batch_shape = '(2,3,1)'
  []
  [error_1]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data.csv'
    column_names = 'col1 col2'
    column_indices = '0 1'
  []
  [error_2]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data.csv'
    column_names = 'col1 col2'
    no_header = true
  []
  [error_3]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data.csv'
    column_names = 'col1 col4'
  []
  [error_4]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data.csv'
    column_indices = '0 3 1'
  []
  [error_5]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data_non_numeric.csv'
  []
  [error_6]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data_non_numeric.csv'
    column_indices = '0 1'
  []
  [error_7]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data.csv'
    batch_shape = '(5,8)'
  []
[]
