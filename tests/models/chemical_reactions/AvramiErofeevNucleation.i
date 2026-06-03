# Translated from tests/unit/models/chemical_reactions/AvramiErofeevNucleation.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'alpha'
    input_Scalar_values = 'alpha'
    output_Scalar_names = 'out'
    output_Scalar_values = 'out'
    derivative_abs_tol = 1e-7
  []
[]

[Tensors]
  [alpha]
    type = Python
    expr = "Scalar(torch.tensor([0.03, 0.75, 0.9], dtype=torch.float64))"
  []
  [out]
    type = Python
    expr = "Scalar(torch.tensor([4.592994468e-5, 0.429675464, 0.69373113], dtype=torch.float64))"
  []
[]

[Models]
  [model]
    type = AvramiErofeevNucleation
    coef = 0.7
    order = 2.75
    conversion_degree = 'alpha'
    reaction_rate = 'out'
  []
[]
