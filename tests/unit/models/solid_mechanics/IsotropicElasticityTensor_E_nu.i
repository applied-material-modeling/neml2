[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'params/E params/nu'
    input_Scalar_values = '100000.0 0.3'
    output_SSR4_names = 'parameters/p'
    output_SSR4_values = 'p_correct'
  []
[]

[Tensors]
  [p_correct]
    type = SSR4
    values = "134615.3846153846 57692.30769230767 57692.30769230767 0.0 0.0 0.0 57692.30769230767 "
             "134615.3846153846 57692.30769230767 0.0 0.0 0.0 57692.30769230767 57692.30769230767 "
             "134615.3846153846 0.0 0.0 0.0 0.0 0.0 0.0 76923.07692307692 0.0 0.0 0.0 0.0 0.0 0.0 "
             "76923.07692307692 0.0 0.0 0.0 0.0 0.0 0.0 76923.07692307692"
  []
[]

[Models]
  [E]
    type = ScalarInputParameter
    from = 'params/E'
  []
  [nu]
    type = ScalarInputParameter
    from = 'params/nu'
  []
  [p]
    type = IsotropicElasticityTensor
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    coefficients = 'E nu'
  []
  [model]
    type = ComposedModel
    models = 'p'
  []
[]
