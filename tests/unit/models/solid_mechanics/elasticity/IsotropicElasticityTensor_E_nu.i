[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/E state/nu'
    input_Scalar_values = '100000.0 0.3'
    output_SSR4_names = 'parameters/C'
    output_SSR4_values = 'C_correct'
  []
[]

[Tensors]
  [C_correct]
    type = SSR4
    values = "134615.3846153846 57692.30769230767 57692.30769230767 0.0 0.0 0.0 57692.30769230767 "
             "134615.3846153846 57692.30769230767 0.0 0.0 0.0 57692.30769230767 57692.30769230767 "
             "134615.3846153846 0.0 0.0 0.0 0.0 0.0 0.0 76923.07692307692 0.0 0.0 0.0 0.0 0.0 0.0 "
             "76923.07692307692 0.0 0.0 0.0 0.0 0.0 0.0 76923.07692307692"
  []
[]

[Models]
  [C]
    type = IsotropicElasticityTensor
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    coefficients = 'state/E state/nu'
  []
  [model]
    type = ComposedModel
    models = 'C'
  []
[]
