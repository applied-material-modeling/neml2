[Models]
  [alpha]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = '300 400 500'
    ordinate = '1e-5 1.5e-5 1.8e-5'
  []
  [K]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = '300 350 400 450'
    ordinate = '1.4e5 1.35e5 1.32e5 1.25e5'
  []
  [G]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = '300 500'
    ordinate = '7.8e4 7e4'
  []
  [eq1]
    type = ThermalEigenstrain
    reference_temperature = '300'
    CTE = 'alpha'
    eigenstrain = 'forces/Eg'
  []
  [eq2]
    type = SR2LinearCombination
    from_var = 'forces/E forces/Eg'
    to_var = 'forces/Ee'
    coefficients = '1 -1'
  []
  [eq3]
    type = LinearIsotropicElasticity
    strain = 'forces/Ee'
    stress = 'state/S'
    coefficient_types = 'BULK_MODULUS SHEAR_MODULUS'
    coefficients = 'K G'
  []
  [eq]
    type = ComposedModel
    models = 'eq1 eq2 eq3'
  []
[]
