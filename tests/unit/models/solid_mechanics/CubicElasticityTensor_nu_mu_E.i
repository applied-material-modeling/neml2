[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'params/E params/nu params/mu'
    input_Scalar_values = '100000.0 0.3 60000.0'
    output_SSR4_names = 'parameters/p'
    output_SSR4_values = 'p_correct'
  []
[]

[Tensors]
  [p_correct]
    type = SSR4
    values = "134615.38461538462 57692.307692307695 57692.307692307695 0.0 0.0 0.0 "
             "57692.307692307695 134615.38461538462 57692.307692307695 0.0 0.0 0.0 "
             "57692.307692307695 57692.307692307695 134615.38461538462 0.0 0.0 0.0 0.0 0.0 0.0 "
             "120000.0 0.0 0.0 0.0 0.0 0.0 0.0 120000.0 0.0 0.0 0.0 0.0 0.0 0.0 120000.0"
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
  [mu]
    type = ScalarInputParameter
    from = 'params/mu'
  []
  [p]
    type = CubicElasticityTensor
    coefficient_types = 'poissons_ratio shear_modulus youngs_modulus'
    coefficients = 'nu mu E'
  []
  [model]
    type = ComposedModel
    models = 'p'
  []
[]
