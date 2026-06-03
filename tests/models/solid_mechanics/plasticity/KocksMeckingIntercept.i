# Translated from tests/unit/models/solid_mechanics/plasticity/KocksMeckingIntercept.i.
# The native ModelUnitTest requires at least one JVP comparison, so promote
# A, B, C to runtime inputs (mode-4 ``declare_typed_parameter``) by giving
# each a bare variable name that does NOT match a ``[Tensors]`` block --
# otherwise the mode-2 lookup would resolve the parameter statically and no
# input would be added. The driver then supplies the three inputs through
# input_Scalar_values, which reference the underlying ``[Tensors]`` data.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'p'
    input_Scalar_names = 'A_in B_in C_in'
    input_Scalar_values = 'A_data B_data C_data'
    output_Scalar_names = 'intercept'
    output_Scalar_values = 'p_correct'
  []
[]

[Models]
  [p]
    type = KocksMeckingIntercept
    A = 'A_in'
    B = 'B_in'
    C = 'C_in'
  []
[]

[Tensors]
  [A_data]
    type = Python
    expr = "Scalar(torch.linspace(-2.0, -3.0, 5, dtype=torch.float64))"
  []
  [B_data]
    type = Python
    expr = "Scalar(torch.linspace(-4.0, -7.0, 5, dtype=torch.float64))"
  []
  [C_data]
    type = Python
    expr = "Scalar(torch.linspace(-5.0, -8.0, 5, dtype=torch.float64))"
  []
  [p_correct]
    type = Python
    expr = "Scalar(torch.tensor([0.5, 0.44444444, 0.4, 0.36363636, 0.33333333], dtype=torch.float64))"
  []
[]
