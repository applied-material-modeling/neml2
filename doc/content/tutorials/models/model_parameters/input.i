# Same linear-isotropic-elasticity model as the previous tutorial.
# Re-used here so the focus stays on parameter access / mutation rather
# than on a new physical setup.
[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients      = '200e3          0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
