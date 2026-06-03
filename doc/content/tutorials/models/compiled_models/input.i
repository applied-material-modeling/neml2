# Linear isotropic elasticity model used as the compilation target.
# E  = 200 GPa, nu = 0.3 — same model as the "hello world" tutorial,
# repeated here so this page is self-contained.
[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients      = '200e3          0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
