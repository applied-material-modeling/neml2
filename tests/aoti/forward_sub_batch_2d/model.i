# forward_sub_batch_2d: forward leaf with a TWO-axis sub-batch input.
# Exercises the multi-dim sub-batch path: every reshape / view needs to
# preserve the two trailing sub axes between dyn and base. Most sub-batch
# bugs only surface with 2+ axes since a single-axis case can accidentally
# work via right-aligned broadcasting.
#
# `[Settings]/example_batch_shape = '(2; 3, 4)'` declares the trace
# shape as dyn=(2,), sub=(3, 4). Strain input shape becomes
# (2, 3, 4, 6) and stress output matches.

[Models]
  [model]
    type = LinearIsotropicElasticity
    strain = 'strain'
    stress = 'stress'
    coefficients = '100 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]

[Settings]
  example_batch_shape = '(2; 3, 4)'
[]
