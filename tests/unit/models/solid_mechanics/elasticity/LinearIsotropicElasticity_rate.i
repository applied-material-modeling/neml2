[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'state/internal/Ee_rate'
    input_SR2_values = 'Ee_rate'
    output_SR2_names = 'state/S_rate'
    output_SR2_values = 'S_rate'
  []
[]

[Tensors]
  [Ee_rate]
    type = FillSR2
    values = '0.09 0.04 -0.02'
  []
  [S_rate]
    type = FillSR2
    values = '13.2692 9.4231 4.8077'
  []
[]

[Models]
  [model]
    type = LinearIsotropicElasticity
    coefficients = '100 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    rate_form = true
  []
[]
