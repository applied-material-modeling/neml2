[Tensors]
  [K_x]
    type = Scalar
    values = '300 350 400 450'
    batch_shape = '(4)'
  []
  [K_y]
    type = Scalar
    values = '1.4e5 1.35e5 1.32e5 1.25e5'
    batch_shape = '(4)'
  []
  [G_x]
    type = Scalar
    values = '300 500'
    batch_shape = '(2)'
  []
  [G_y]
    type = Scalar
    values = '7.8e4 7e4'
    batch_shape = '(2)'
  []
[]

[Models]
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
    CTE = 'forces/alpha'
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
