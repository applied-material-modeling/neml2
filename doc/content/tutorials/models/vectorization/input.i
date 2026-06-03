# Reuses the hello-world elasticity model from the
# "Running your first model" tutorial. Linear isotropic elasticity
# is a trivial map (SR2 -> SR2), which keeps the focus on *how* we
# feed in a batch of strains rather than what the model is doing.
[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients      = '200e3          0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
