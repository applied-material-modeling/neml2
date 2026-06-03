# Translated from tests/unit/models/finite_volume/SemiInfiniteCoordinateTransform.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/x'
    input_Scalar_values = 'x'
    output_Scalar_names = 'state/x_hat'
    output_Scalar_values = 'x_hat'
  []
[]

[Tensors]
  [x]
    type = Python
    expr = 'Scalar(2.0)'
  []
  [x_hat]
    type = Python
    expr = 'Scalar(0.6666666666666666)'
  []
[]

[Models]
  [model]
    type = SemiInfiniteCoordinateTransform
    coordinate = 'state/x'
    shift = 1.0
    transformed_coordinate = 'state/x_hat'
  []
[]
