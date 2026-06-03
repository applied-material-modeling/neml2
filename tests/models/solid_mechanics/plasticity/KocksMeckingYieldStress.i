# Translated from tests/unit/models/solid_mechanics/plasticity/KocksMeckingYieldStress.i.
# The C++ fixture statically binds both parameters via [Tensors] cross-refs.
# The native ModelUnitTest needs at least one JVP comparison, so we promote
# both parameters to runtime inputs (mode-4 declare_typed_parameter: a bare
# variable name with no matching [Tensors] entry) and feed their values via
# the driver's input_Scalar_values.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'p'
    input_Scalar_names = 'C_in mu_in'
    input_Scalar_values = 'C mu'
    output_Scalar_names = 'yield_stress'
    output_Scalar_values = 'p_correct'
  []
[]

[Models]
  [p]
    type = KocksMeckingYieldStress
    C = 'C_in'
    shear_modulus = 'mu_in'
  []
[]

[Tensors]
  [mu]
    type = Python
    expr = "Scalar(torch.linspace(50000.0, 100000.0, 5, dtype=torch.float64))"
  []
  [C]
    type = Python
    expr = "Scalar(torch.linspace(-3.5, -5.5, 5, dtype=torch.float64))"
  []
  [p_correct]
    type = Python
    expr = "Scalar(torch.tensor([1509.86917112, 1144.72743055, 833.17474037, 589.57036242, 408.67714385], dtype=torch.float64))"
  []
[]
