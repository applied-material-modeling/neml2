# neml2
# Coalesced branch: f = 0.6 sits above f_c = 0.5, so the GTN acceleration is
# active and the slope is kappa rather than 1. Pinning the point here (rather
# than below f_c, where the map is the identity and a broken kappa would pass)
# makes the parameters load-bearing.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'cavity_fraction'
    # kappa = (0.99 - 0.5) / (0.8 - 0.5) = 1.6333333333
    # f* = 0.5 + 1.6333333333 * (0.6 - 0.5) = 0.6633333333
    input_Scalar_values = '0.6'
    output_Scalar_names = 'effective_cavity_fraction'
    output_Scalar_values = '0.6633333333'
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
