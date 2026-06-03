# Same linear-isotropic-elastic "elasticity" model used in the earlier
# tutorials, repeated here so the autograd page is self-contained.
#   E  = 200 GPa
#   nu = 0.3
[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients      = '200e3          0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
