[Tensors]
  [a]
    type = CSVScalar
    csv_file = 'user_tensors/CSVPrimitiveTensor.csv'
    component_names = 'var'
    batch_shape = '(4)'
  []
  [b]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor.csv'
    component_names = 'disp_x disp_y disp_z'
    batch_shape = '(4)'
  []
  [c]
    type = CSVVec
    csv_file = 'user_tensors/CSVPrimitiveTensor.csv'
    component_names = 'disp_x disp_y disp_z'
    batch_shape = '(2,2)'
  []
  # [d]
  #   type = CSVSR2
  #   csv_file = 'user_tensors/CSVPrimitiveTensor.csv'
  #   component_names = 's_xx s_yy s_zz s_yz s_xz s_xy'
  #   batch_shape = '(4)'
  # []
[]