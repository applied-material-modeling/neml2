# neml2
# New cavities arrive at the current mean size, so fdot = (f/N) Ndot.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'cavity_fraction cavity_density cavity_density_rate'
    # fdot = 0.6 * 11.1386138615 / 3.0 = 2.2277227723
    input_Scalar_values = '0.6 3.0 11.1386138615'
    output_Scalar_names = 'cavity_fraction_nucleation_rate'
    output_Scalar_values = '2.2277227723'
    value_abs_tol = 1e-8
  []
[]

[Models]
  [model]
    type = CavityNucleationDamageRate
  []
[]
