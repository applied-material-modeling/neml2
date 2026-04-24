[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'ri ro R_l R_s'
    input_Scalar_values = '0.5 0.8 0.2 0.3'
    output_Scalar_names = 'alpha_rate'
    output_Scalar_values = '0.000309677419'
  []
[]

[Models]
  [model]
    type = DiffusionLimitedReaction
    diffusion_coefficient = 1e-3
    molar_volume = 1
    product_inner_radius = 'ri'
    solid_inner_radius = 'ro'
    liquid_reactivity = 'R_l'
    solid_reactivity = 'R_s'
    reaction_rate = 'alpha_rate'
  []
[]
