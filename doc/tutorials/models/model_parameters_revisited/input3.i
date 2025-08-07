[Tensors]
  [alpha_x]
    type = Scalar
    values = '300 400 500'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [alpha_y]
    type = Scalar
    values = '1e-5 1.5e-5 1.8e-5'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [K_x]
    type = Scalar
    values = '300 350 400 450'
    batch_shape = '(4)'
    intermediate_dimension = 1
  []
  [K_y]
    type = Scalar
    values = '1.4e5 1.35e5 1.32e5 1.25e5'
    batch_shape = '(4)'
    intermediate_dimension = 1
  []
  [G_x]
    type = Scalar
    values = '300 500'
    batch_shape = '(2)'
    intermediate_dimension = 1
  []
  [G_y]
    type = Scalar
    values = '7.8e4 7e4'
    batch_shape = '(2)'
    intermediate_dimension = 1
  []
[]

[Models]
  [alpha]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = 'alpha_x'
    ordinate = 'alpha_y'
  []
  [K]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = 'K_x'
    ordinate = 'K_y'
  []
  [G]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = 'G_x'
    ordinate = 'G_y'
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
