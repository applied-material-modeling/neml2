[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'phi_s phi_p'
    input_Scalar_values = '0.1 0.2'
    output_Scalar_names = 'ri ro'
    output_Scalar_values = '0.83666 0.948683'
  []
[]

[Models]
  [model]
    type = CylindricalChannelGeometry
    solid_fraction = 'phi_s'
    product_fraction = 'phi_p'
    inner_radius = 'ri'
    outer_radius = 'ro'
  []
[]
