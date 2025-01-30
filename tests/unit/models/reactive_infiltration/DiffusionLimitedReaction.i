[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/ri state/ro state/R_l state/R_s'
    input_Scalar_values = '0.5 0.8 0.2 0.3'
    output_Scalar_names = 'state/alpha_rate'
    output_Scalar_values = '0.000503226'
  []
[]

[Models]
  [model]
    type = DiffusionLimitedReaction
    diffusion_coefficient = 1e-3
    molar_volume = 1
    product_inner_radius = 'state/ri'
    solid_inner_radius = 'state/ro'
    liquid_reactivity = 'state/R_l'
    solid_reactivity = 'state/R_s'
    reaction_rate = 'state/alpha_rate'
  []
[]
