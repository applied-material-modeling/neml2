[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/ri state/ro state/R_l state/R_s'
    input_Scalar_values = '0.5 0.8 0.2 0.3'
    output_Scalar_names = 'state/phi_p_rate state/phi_s_rate'
    output_Scalar_values = '0.00150968 -0.00100645'
  []
[]

[Models]
  [model]
    type = DiffusionLimitedReaction
    diffusion_coefficient = 1e-3
    liquid_molar_volume = 1
    solid_molar_volume = 2
  []
[]
