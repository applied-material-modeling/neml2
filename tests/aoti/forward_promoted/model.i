# forward_promoted: same as forward_single. The test driver compiles this
# with `--parameter E --parameter nu` to exercise the explicit-promotion path
# through the PromotedParam machinery.

[Models]
  [model]
    type = LinearIsotropicElasticity
    strain = 'strain'
    stress = 'stress'
    coefficients = '100 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
