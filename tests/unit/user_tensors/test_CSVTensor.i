[Tensors]
  [a]
    type = ScalarCSVTensor
    csv_file = '/home/jiw/packages/neml2/tests/unit/user_tensors/ScalarCSVTensorData.csv'
    batch_shape = '(3,2)'
  []
  [b]
    type = ScalarCSVTensor
    csv_file = '/home/jiw/packages/neml2/tests/unit/user_tensors/ScalarCSVTensorData.csv'
    batch_shape = '(2,2)'
    csv_columns = 'col1 col2'
  []
  [c]
    type = SR2CSVTensor
    csv_file = '/home/jiw/packages/neml2/tests/unit/user_tensors/SR2CSVTensorData.csv'
    batch_shape = '(3,2)'
  []
[]
