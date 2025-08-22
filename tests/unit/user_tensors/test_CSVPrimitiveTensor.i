[Tensors]
  [all_columns_vector]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_vec.csv'
  []
  [all_columns_SR2]
    type = CSVSR2
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_SR2.csv'
  []
  [scalar]
    type = CSVScalar
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_all.csv'
    column_names = 'var'
  []
  [SR2]
    type = CSVSR2
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_all.csv'
    column_names = 's_xx s_yy s_zz s_yz s_xz s_xy'
  []
  [delimiter]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_vec_space.csv'
    delimiter = 'SPACE'
  []
  [starting_row]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_vec_comment.csv'
    starting_row = 2
  []
  [no_header]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_vec_no_header.csv'
    no_header = true
  []
  [starting_row_no_header]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_vec_no_header_comment.csv'
    starting_row = 2
    no_header = true
  []
  [no_header_indices]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_vec_no_header.csv'
    column_indices = '2 1 0'
    no_header = true
  []
  [batch_shape]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_vec.csv'
    batch_shape = '(2,2)'
  []
  [error_col_name_col_ind]
    type = CSVScalar
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_all.csv'
    column_names = 'var'
    column_indices = '0'
  []
  [error_col_name_header]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_vec.csv'
    column_names = 'disp_x disp_y disp_z'
    no_header = true
  []
  [error_batch_shape]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_vec.csv'
    batch_shape = '(2,3)'
  []
  [error_starting_row]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_vec.csv'
    starting_row = -1
  []
  [error_col_name]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_vec.csv'
    column_names = 'disp_x disp_w disp_z'
  []
  [error_col_ind]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_vec.csv'
    column_indices = '0 1 3'
  []
  [error_non_numeric_read_all_header]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_vec_non_numeric.csv'
  []
  [error_non_numeric_read_all_no_header]
    type = CSVSR2
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_SR2_non_numeric_no_header.csv'
    no_header = true
  []
  [error_non_numeric_read_ind_no_header]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_all_non_numeric_no_header.csv'
    column_indices = '0 1 2'
    no_header = true
  []
  [error_non_numeric_read_ind_header]
    type = CSVSR2
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_all_non_numeric.csv'
    column_names = 's_xx s_yy s_zz s_yz s_xz s_xy'
  []
  [error_col_name_component_mismatch]
    type = CSVScalar
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_all.csv'
    column_names = 'var1 var2'
  []
  [error_col_ind_component_mismatch]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_all.csv'
    column_indices = '0 1'
  []
  [error_col_no_component_mismatch]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_vec_incomplete.csv'
  []
  [error_col_no_component_mismatch_SR2]
    type = CSVSR2
    csv_file = 'user_tensors/CSVPrimitiveTensor/data_SR2_incomplete.csv'
  []
[]
