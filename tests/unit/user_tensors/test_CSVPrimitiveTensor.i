[Tensors]
  [all_columns_vector]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data.csv'
  []
  [scalar]
    type = CSVScalar
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_supported_types.csv'
    column_names = 'var'
  []
  [all_columns_SR2]
    type = CSVSR2
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_SR2.csv'
  []
  [delimiter]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_space.csv'
    delimiter = 'SPACE'
  []
  [starting_row]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_comment.csv'
    starting_row = 2
  []
  [no_header]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_no_header.csv'
    no_header = true
  []
  [starting_row_no_header]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_no_header_comment.csv'
    starting_row = 2
    no_header = true
  []
  [no_header_indices]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_no_header.csv'
    column_indices = '2 1 0'
    no_header = true
  []
  [error_starting_row]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data.csv'
    starting_row = -1
  []
  [error_col_name_component_mismatch]
    type = CSVSR2
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_SR2.csv'
    column_names = 's_xx s_yy'
  []
  [error_col_ind_component_mismatch]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data.csv'
    column_indices = '0 1'
  []
  [error_col_name_col_ind]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data.csv'
    column_names = 'disp_x disp_y disp_z'
    column_indices = '0 1 2'
  []
  [error_col_name_header]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data.csv'
    column_names = 'disp_x disp_y disp_z'
    no_header = true
  []
  [error_col_name]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data.csv'
    column_names = 'disp_x disp_w disp_z'
  []
  [error_non_numeric_read_all_no_header]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_non_numeric_no_header.csv'
    no_header = true
  []
  [error_non_numeric_read_all_header]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_non_numeric.csv'
  []
  [error_ind_out_of_bounds]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data.csv'
    column_indices = '0 1 3'
  []
  [error_non_numeric_read_ind_no_header]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_non_numeric_no_header.csv'
    column_indices = '0 1 2'
    no_header = true
  []
  [error_non_numeric_read_ind_header]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_non_numeric.csv'
    column_names = 'disp_x disp_y disp_z'
  []
  [error_col_no_component_mismatch]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_incomplete.csv'
  []
[]
