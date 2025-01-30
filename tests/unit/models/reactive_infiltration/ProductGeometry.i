[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/phi_s state/phi_p'
    input_Scalar_values = '0.1 0.2'
    output_Scalar_names = 'state/ri state/ro'
    output_Scalar_values = '0.83666 0.948683'
  []
[]

[Models]
  [model]
    type = ProductGeometry
    solid_fraction = 'state/phi_s'
    product_fraction = 'state/phi_p'
    inner_radius = 'state/ri'
    outer_radius = 'state/ro'
  []
[]
