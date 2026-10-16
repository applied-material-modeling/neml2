# neml2
# Compression switches nucleation off through the Macaulay bracket. Exponent
# 0.5 is deliberate: the masked-off base would make the x^(beta-1) derivative
# singular at zero if the mask were applied after the power instead of before.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'normal_traction strain_rate'
    input_Scalar_values = '-25.0 14.8514851486'
    output_Scalar_names = 'cavity_density_rate'
    output_Scalar_values = '0.0'
    value_abs_tol = 1e-12
  []
[]

[Models]
  [model]
    type = CavityNucleationRate
    nucleation_coefficient = 3.0
    reference_stress = 50.0
    exponent = 0.5
  []
[]
