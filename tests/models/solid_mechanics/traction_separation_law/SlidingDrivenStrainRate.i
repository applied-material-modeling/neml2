# neml2
# Local nucleation driver. The sliding rate is negative here so the absolute
# value is exercised and the sign of the pushforward is pinned.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'viscous_sliding_rate'
    # edot = |-7.4257425743| / 0.5 = 14.8514851486
    input_Scalar_values = '-7.4257425743'
    output_Scalar_names = 'strain_rate'
    output_Scalar_values = '14.8514851486'
    value_abs_tol = 1e-8
  []
[]

[Models]
  [model]
    type = SlidingDrivenStrainRate
    length_scale = 0.5
  []
[]
