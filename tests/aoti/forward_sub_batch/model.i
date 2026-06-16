# forward_sub_batch: a single forward leaf with a per-site sub-batched
# input. Tests that typed-wrapper inputs carry their `sub_batch_ndim`
# through `torch.export` and that the AOTI shim's per-input sub-batch
# split survives the .pt2 round-trip.
#
# `[Settings]/example_batch_shape = '(2; 5)'` declares the trace shape
# as dyn=(2,), sub=(5,). The strain input thus has shape (2, 5, 6); the
# stress output has the same. The test harness reads this declaration
# and constructs runtime inputs with matching shapes so the comparison
# is meaningful.

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
  example_batch_shape = '(2; 5)'
[]
