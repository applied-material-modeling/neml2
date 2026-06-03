# Translated from tests/unit/models/solid_mechanics/elasticity/CubicElasticityTensor_E_nu_mu.i.
# E, nu, and mu are promoted to runtime inputs (mode 4) so the ModelUnitTest
# driver can vary them; the resolved output variable defaults to the HIT block
# name "C".
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'C'
    input_Scalar_names = 'E nu mu'
    input_Scalar_values = '100000.0 0.3 60000.0'
    output_SSR4_names = 'C'
    output_SSR4_values = 'C_correct'
  []
[]

[Tensors]
  [C_correct]
    type = Python
    expr = '''
      SSR4(torch.tensor([
          [134615.38461538462,  57692.307692307695, 57692.307692307695, 0.0, 0.0, 0.0],
          [ 57692.307692307695, 134615.38461538462, 57692.307692307695, 0.0, 0.0, 0.0],
          [ 57692.307692307695, 57692.307692307695, 134615.38461538462, 0.0, 0.0, 0.0],
          [   0.0,                0.0,               0.0,            120000.0, 0.0, 0.0],
          [   0.0,                0.0,               0.0,            0.0, 120000.0, 0.0],
          [   0.0,                0.0,               0.0,            0.0, 0.0, 120000.0],
      ], dtype=torch.float64))
    '''
  []
[]

[Models]
  [C]
    type = CubicElasticityTensor
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO SHEAR_MODULUS'
    coefficients = 'E nu mu'
  []
[]
