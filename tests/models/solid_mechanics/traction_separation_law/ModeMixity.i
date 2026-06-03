# Translated from tests/unit/models/solid_mechanics/traction_separation_law/ModeMixity.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'normal_separation tangential_separation'
    input_Scalar_values = '0.5 0.5'
    output_Scalar_names = 'mode_mixity'
    # beta = ds / dn = 0.5 / 0.5 = 1.0  (opening branch)
    output_Scalar_values = '1.0'
    derivative_abs_tol = 1e-6
  []
[]

[Models]
  [model]
    type = ModeMixity
  []
[]
