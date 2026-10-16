# neml2
# Compressive branch: the Macaulay bracket must switch sintering off outright,
# so the rate is exactly zero and the normal-traction partial is zero too.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'normal_traction effective_cavity_fraction'
    input_Scalar_values = '-25.0 0.6633333333'
    output_Scalar_names = 'cavity_volume_rate'
    output_Scalar_values = '0.0'
    value_abs_tol = 1e-12
  []
[]

[Models]
  [model]
    type = DiffusiveCavityGrowthRate
    diffusivity = 1.0e-6
  []
[]
