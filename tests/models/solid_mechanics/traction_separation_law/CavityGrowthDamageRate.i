# neml2
# Geometric conversion from a single-cavity volume rate to an area-fraction
# rate. The pinned value is cross-checked against the closed form given in the
# issue, 4 pi^(3/2) D N^(3/2) <Tn> / (h sqrt(f) q(f*)), in
# tests/unit/test_sham_needleman.py.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'cavity_fraction cavity_density cavity_volume_rate'
    # geo = sqrt(pi) * 3^1.5 / (2 * 0.61 * sqrt(0.6)) = 9.7459624654
    # fdot = 9.7459624654 * 0.0183303370 = 0.1786453892
    input_Scalar_values = '0.6 3.0 0.0183303370'
    output_Scalar_names = 'cavity_fraction_growth_rate'
    output_Scalar_values = '0.1786453892'
    value_abs_tol = 1e-8
  []
[]

[Models]
  [model]
    type = CavityGrowthDamageRate
    cavity_shape_factor = 0.61
  []
[]
