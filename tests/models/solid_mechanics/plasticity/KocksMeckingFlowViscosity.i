# Translated from tests/unit/models/solid_mechanics/plasticity/KocksMeckingFlowViscosity.i.
# The C++ fixture statically binds A / B / shear_modulus via [Tensors] LinspaceScalar
# cross-refs and fixes the temperature input at 1000. Native ModelUnitTest needs
# at least one JVP comparison, so we promote all three parameters to runtime inputs
# (mode-4 declare_typed_parameter: bare variable names with no matching [Tensors]
# entry) and feed their values via the driver's input_Scalar_values.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'p'
    input_Scalar_names = 'temperature A_in B_in mu_in'
    input_Scalar_values = '1000 A B mu'
    output_Scalar_names = 'p'
    output_Scalar_values = 'p_correct'
  []
[]

[Models]
  [p]
    type = KocksMeckingFlowViscosity
    shear_modulus = 'mu_in'
    A = 'A_in'
    B = 'B_in'
    eps0 = 1e10
    k = 1.38064e-20
    b = 2.019e-7
    temperature = 'temperature'
  []
[]

[Tensors]
  [mu]
    type = Python
    expr = "Scalar(torch.linspace(50000.0, 100000.0, 5, dtype=torch.float64))"
  []
  [A]
    type = Python
    expr = "Scalar(torch.linspace(-3.5, -5.5, 5, dtype=torch.float64))"
  []
  [B]
    type = Python
    expr = "Scalar(torch.linspace(-1.5, -3.0, 5, dtype=torch.float64))"
  []
  [p_correct]
    type = Python
    expr = "Scalar(torch.tensor([746.88551856, 809.01337039, 778.71385139, 697.25850376, 594.93909239], dtype=torch.float64))"
  []
[]
