# neml2
# Smeared cavity volume as a normal opening, measured from the initial state.
# f0 and N0 are nonzero so the offset term is load-bearing.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'cavity_fraction cavity_density'
    # c   = 4 * 0.61 / (3 sqrt(pi)) = 0.4589597020
    # NV  = c * 0.6^1.5 / sqrt(3.0) = 0.1231528940
    # NV0 = c * 0.01^1.5 / sqrt(2.0) = 0.0003245000
    # opening = 0.1228043940
    input_Scalar_values = '0.6 3.0'
    output_Scalar_names = 'viscous_normal_opening'
    output_Scalar_values = '0.1228043940'
    value_abs_tol = 1e-9
  []
[]

[Models]
  [model]
    type = CavityVolumeOpening
    cavity_shape_factor = 0.61
    initial_cavity_fraction = 0.01
    initial_cavity_density = 2.0
  []
[]
