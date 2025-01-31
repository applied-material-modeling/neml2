[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/ri state/ro state/R_l state/R_s'
    input_Scalar_values = '0.5 0.8 0.2 0.3'
    output_Scalar_names = 'state/alpha_rate '
    output_Scalar_values = '0.000503226'
  []
[]

[Models]
  [model]
    type = DiffusionLimitedReaction
    diffusion_coefficient = 1e-3
    molar_volume = 1
  []
[]
