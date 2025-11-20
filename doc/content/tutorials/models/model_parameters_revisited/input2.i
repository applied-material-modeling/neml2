[Tensors]
  [alpha]
    type = Scalar
    values = '1e-6 2e-6 1e-5 5e-7'
    batch_shape = (2,2)
  []
[]

[Models]
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
    coefficients = '1.4e5 7.8e4'
  []
  [eq]
    type = ComposedModel
    models = 'eq1 eq2 eq3'
  []
[]
