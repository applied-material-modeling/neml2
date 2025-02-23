[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/E state/nu'
    input_Scalar_values = '100 0.3'
    input_SR2_names = 'state/internal/Ee'
    input_SR2_values = 'Ee'
    output_SR2_names = 'state/S'
    output_SR2_values = 'S'
  []
[]

[Tensors]
  [Ee]
    type = FillSR2
    values = '0.09 0.04 -0.02'
  []
  [S]
    type = FillSR2
    values = '13.2692 9.4231 4.8077'
  []
[]

[Models]
  [model0]
    type = LinearIsotropicElasticity
    coefficients = 'state/E state/nu'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
