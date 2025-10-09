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
    csv_file = 'user_tensors/MultiColumnCSVScalar/data_space.csv'
    delimiter = 'SPACE'
  []
  [starting_row]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data_comment.csv'
    starting_row = 2
  []
  [no_header]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data_no_header.csv'
    no_header = true
  []
  [starting_row_no_header]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data_no_header_comment.csv'
    starting_row = 2
    no_header = true
  []
  [no_header_indices]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data_no_header.csv'
    column_indices = '2 0'
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
  [error_starting_row]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data.csv'
    starting_row = -1
  []
  [error_col_name_col_ind]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data.csv'
    column_names = 'col1 col2'
    column_indices = '0 1'
  []
  [error_col_name_header]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data.csv'
    column_names = 'col1 col2'
    no_header = true
  []
  [error_col_name]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data.csv'
    column_names = 'col1 col4'
  []
  [error_non_numeric_read_all_no_header]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data_non_numeric_no_header.csv'
    no_header = true
  []
  [error_non_numeric_read_all_header]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data_non_numeric.csv'
  []
  [error_non_numeric_read_ind_no_header]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data_non_numeric_no_header.csv'
    column_indices = '0 1'
    no_header = true
  []
  [error_non_numeric_read_ind_header]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data_non_numeric.csv'
    column_names = 'col1 col2'
  []
  [error_ind_out_of_bounds]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data.csv'
    column_indices = '0 3 1'
  []
  [error_batch_shape]
    type = MultiColumnCSVScalar
    csv_file = 'user_tensors/MultiColumnCSVScalar/data.csv'
    batch_shape = '(5,8)'
  []

[]
