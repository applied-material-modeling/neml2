# Minimal hello-world NEML2 input file.
# A single linear isotropic elastic model named "elasticity":
#   E  = 200 GPa
#   nu = 0.3
# Maps a symmetric strain tensor (SR2) to a symmetric stress tensor (SR2).
[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients      = '200e3          0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
