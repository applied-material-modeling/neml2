# A thermo-elastic constitutive law composed from three pieces:
#
#   eq1: eigenstrain  = alpha * (T - T0) * I
#   eq2: elastic_strain = strain - eigenstrain
#   eq3: stress = 3 K vol(elastic_strain) + 2 G dev(elastic_strain)
#
# Every parameter is set with a plain numeric literal.
[Models]
  [eq1]
    type = ThermalEigenstrain
    reference_temperature = '300'
    CTE                   = '1e-6'
  []
  [eq2]
    type    = SR2LinearCombination
    from    = 'strain eigenstrain'
    to      = 'elastic_strain'
    weights = '1 -1'
  []
  [eq3]
    type              = LinearIsotropicElasticity
    strain            = 'elastic_strain'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    coefficients      = '2e5            0.3'
  []
  [eq]
    type   = ComposedModel
    models = 'eq1 eq2 eq3'
  []
[]
