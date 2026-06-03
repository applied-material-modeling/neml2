# forward_single: a one-leaf forward-only model. Smoke test for the
# bake-by-default path on the simplest possible shape.

[Models]
  [model]
    type = LinearIsotropicElasticity
    strain = 'strain'
    stress = 'stress'
    coefficients = '100 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
