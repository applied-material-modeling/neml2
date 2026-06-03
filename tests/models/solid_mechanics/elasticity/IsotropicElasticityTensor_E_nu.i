# Translated from tests/unit/models/solid_mechanics/elasticity/IsotropicElasticityTensor_E_nu.i.
# E and nu are promoted to runtime inputs (mode 4) so the ModelUnitTest driver
# can vary them; the resolved output variable defaults to the HIT block name "C".
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'C'
    input_Scalar_names = 'E nu'
    input_Scalar_values = '100000.0 0.3'
    output_SSR4_names = 'C'
    output_SSR4_values = 'C_correct'
  []
[]

[Tensors]
  [C_correct]
    type = Python
    expr = '''
      SSR4(torch.tensor([
          [134615.3846153846,  57692.30769230767, 57692.30769230767, 0.0, 0.0, 0.0],
          [ 57692.30769230767, 134615.3846153846, 57692.30769230767, 0.0, 0.0, 0.0],
          [ 57692.30769230767, 57692.30769230767, 134615.3846153846, 0.0, 0.0, 0.0],
          [   0.0,                0.0,               0.0,            76923.07692307692, 0.0, 0.0],
          [   0.0,                0.0,               0.0,            0.0, 76923.07692307692, 0.0],
          [   0.0,                0.0,               0.0,            0.0, 0.0, 76923.07692307692],
      ], dtype=torch.float64))
    '''
  []
[]

[Models]
  [C]
    type = IsotropicElasticityTensor
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    coefficients = 'E nu'
  []
[]
