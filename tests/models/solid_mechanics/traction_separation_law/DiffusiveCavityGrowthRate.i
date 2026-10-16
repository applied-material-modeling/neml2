# neml2
# Tensile branch of the diffusive growth law. f* = 0.6633333333 puts q well
# into the softening regime (q = 0.0342775221), so a wrong q or a wrong
# prefactor shows up immediately rather than being absorbed by a near-unit
# denominator.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'normal_traction effective_cavity_fraction'
    # q(f*) = -2 ln(0.6633333333) - (3 - 0.6633333333)(1 - 0.6633333333)
    #       = 0.0342775221
    # Vdot = 8 pi * 1e-6 * 25 / 0.0342775221 = 0.0183303370
    input_Scalar_values = '25.0 0.6633333333'
    output_Scalar_names = 'cavity_volume_rate'
    output_Scalar_values = '0.0183303370'
    value_abs_tol = 1e-10
    derivative_abs_tol = 1e-6
  []
[]

[Models]
  [model]
    type = DiffusiveCavityGrowthRate
    diffusivity = 1.0e-6
  []
[]
