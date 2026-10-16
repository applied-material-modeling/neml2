# neml2
# Tensile nucleation with beta = 2.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'normal_traction strain_rate'
    # Ndot = 3 * (25/50)^2 * 14.8514851486 = 11.1386138615
    input_Scalar_values = '25.0 14.8514851486'
    output_Scalar_names = 'cavity_density_rate'
    output_Scalar_values = '11.1386138615'
    value_abs_tol = 1e-8
  []
[]

[Models]
  [model]
    type = CavityNucleationRate
    nucleation_coefficient = 3.0
    reference_stress = 50.0
    exponent = 2.0
  []
[]
