# Translated from tests/unit/models/solid_mechanics/plasticity/KocksMeckingActivationEnergy.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'temperature strain_rate'
    input_Scalar_values = 'T 1.1'
    output_Scalar_names = 'activation_energy'
    output_Scalar_values = 'g_correct'
  []
[]

[Models]
  [model]
    type = KocksMeckingActivationEnergy
    eps0 = 1e10
    k = 1.38064e-20
    b = 2.019e-7
    shear_modulus = 75000.0
  []
[]

[Tensors]
  [T]
    type = Python
    expr = "Scalar(torch.linspace(500.0, 1000.0, 5, dtype=torch.float64))"
  []
  [g_correct]
    type = Python
    expr = "Scalar(torch.tensor([0.25644517, 0.32055647, 0.38466776, 0.44877906, 0.51289035], dtype=torch.float64))"
  []
[]
