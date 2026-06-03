# Linear isotropic elasticity used by the evaluation-device tutorial.
# Same model as the "running your first model" tutorial -- a small,
# device-agnostic forward operator to demonstrate how parameters and
# inputs are moved between CPU and CUDA.
[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients      = '200e3          0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
