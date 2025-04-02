[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/E state/nu state/mu'
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
  [p]
    type = CubicElasticityTensor
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO SHEAR_MODULUS'
    coefficients = 'state/E state/nu state/mu'
  []
  [model]
    type = ComposedModel
    models = 'p'
  []
[]
