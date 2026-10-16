# neml2
# Sub-coalescence branch: f = 0.3 is below f_c, so f* = f identically and the
# slope is 1. Pinned separately from EffectiveCavityFraction.i so the branch
# switch itself is covered, not just the accelerated side.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'cavity_fraction'
    input_Scalar_values = '0.3'
    output_Scalar_names = 'effective_cavity_fraction'
    output_Scalar_values = '0.3'
    value_abs_tol = 1e-8
  []
[]

[Models]
  [model]
    type = EffectiveCavityFraction
    coalescence_fraction = 0.5
    failure_fraction = 0.8
    ultimate_fraction = 0.99
  []
[]
