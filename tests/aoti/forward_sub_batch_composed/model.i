# forward_sub_batch_composed: two forward leaves wired in a ComposedModel
# where the inter-leaf data flow carries a per-site sub-batch. Tests that
# the segment partitioner emits one forward segment whose internal
# topology preserves sub_batch through each leaf boundary -- mirrors
# `forward_composed` but with a non-trivial sub-batch axis.

[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    strain = 'strain'
    stress = 'stress'
    coefficients = '100 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
  [mandel]
    type = IsotropicMandelStress
    cauchy_stress = 'stress'
    mandel_stress = 'mandel_stress'
  []
  [model]
    type = ComposedModel
    models = 'elasticity mandel'
  []
[]

[Settings]
  example_batch_shape = '(2; 5)'
[]
