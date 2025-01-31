[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'forces/alpha state/phi_p'
    input_Scalar_values = '0.5 0.2'
    output_Scalar_names = 'state/phi_l'
    output_Scalar_values = '0.4'
  []
[]

[Models]
  [model]
    type = LiquidFraction
    liquid_molar_volume = 1
    product_molar_volume = 2
  []
[]
