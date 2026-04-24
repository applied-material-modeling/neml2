[Settings]
  parameter_name_separator = '::'
[]

[Models]
  [E1]
    type = ScalarConstantParameter
    value = 1e3
    parameter = 'E1'
  []
  [E2]
    type = ScalarConstantParameter
    value = 2e3
    parameter = 'E2'
  []
  [E3]
    type = ScalarConstantParameter
    value = 3e3
    parameter = 'E3'
  []
  [elasticity1]
    type = LinearIsotropicElasticity
    strain = 'strain'
    stress = 'stress1'
    coefficients = 'E1 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
  [elasticity2]
    type = LinearIsotropicElasticity
    strain = 'strain'
    stress = 'stress2'
    coefficients = 'E2 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
  [elasticity2_another]
    type = LinearIsotropicElasticity
    strain = 'strain'
    stress = 'stress2_another'
    coefficients = 'E2 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
  [elasticity3]
    type = LinearIsotropicElasticity
    strain = 'strain'
    stress = 'stress3_another'
    coefficients = 'E3 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
  [model1]
    type = ComposedModel
    models = 'elasticity1 elasticity2'
  []
  [model2]
    type = ComposedModel
    models = 'elasticity2_another elasticity3'
  []
  [model]
    type = ComposedModel
    models = 'model1 model2'
  []
[]
