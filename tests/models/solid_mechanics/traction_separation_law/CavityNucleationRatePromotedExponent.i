# neml2
# The stress exponent promoted to a runtime input, in tension and at
# Tn / S0 != 1 so that log(Tn / S0) != 0 and the beta tangent is actually
# load-bearing. At Tn / S0 == 1 the partial Ndot * log(x) vanishes for any
# beta, which is exactly the evaluation point that would let a missing
# exponent action pass.
#
# x    = 25 / 50 = 0.5
# Ndot = 3 * 0.5^2 * 14.8514851486 = 11.1386138615
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'normal_traction strain_rate exponent'
    input_Scalar_values = 'Tn edot beta'
    output_Scalar_names = 'cavity_density_rate'
    output_Scalar_values = 'Ndot'
    value_abs_tol = 1e-8
  []
[]

[Tensors]
  [Tn]
    type = Python
    expr = 'Scalar(25.0)'
  []
  [edot]
    type = Python
    expr = 'Scalar(14.8514851486)'
  []
  [beta]
    type = Python
    expr = 'Scalar(2.0)'
  []
  [Ndot]
    type = Python
    expr = 'Scalar(11.13861386145)'
  []
[]

[Models]
  [model]
    type = CavityNucleationRate
    nucleation_coefficient = 3.0
    reference_stress = 50.0
    exponent = 'exponent'
  []
[]
