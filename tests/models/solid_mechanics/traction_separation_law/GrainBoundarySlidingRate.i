# neml2
# Sliding dashpot softened by the cavitated area fraction. f* = 0.6633333333
# so the (1 - f*) ligament factor is 0.3366666667 -- well away from 1, which
# is what makes the softening term load-bearing in the pinned value.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'shear_traction effective_cavity_fraction'
    # rate = 10 / ((1 - 0.6633333333) * 4) = 7.4257425743
    input_Scalar_values = '10.0 0.6633333333'
    output_Scalar_names = 'viscous_sliding_rate'
    output_Scalar_values = '7.4257425743'
    value_abs_tol = 1e-8
  []
[]

[Models]
  [model]
    type = GrainBoundarySlidingRate
    sliding_viscosity = 4.0
  []
[]
