# Translated from tests/unit/models/kwn/SFFKPrecipitationGrowthRate.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/sum state/T'
    input_Scalar_values = 'sum T'
    output_Scalar_names = 'state/R_dot'
    output_Scalar_values = 'R_dot'
  []
[]

[Tensors]
  [R]
    type = Python
    expr = "Scalar(torch.tensor([1.0, 2.0, 4.0]), sub_batch_ndim=1)"
  []
  [sum]
    type = Python
    expr = "Scalar(torch.tensor(0.5))"
  []
  [T]
    type = Python
    expr = "Scalar(torch.tensor(400.0))"
  []
  [R_dot]
    type = Python
    expr = "Scalar(torch.tensor([0.00125, 0.000625, 0.0003125]), sub_batch_ndim=1)"
  []
[]

[Models]
  [model]
    type = SFFKPrecipitationGrowthRate
    radius = 'R'
    projected_diffusivity_sum = 'state/sum'
    gibbs_free_energy_difference = 2.0
    temperature = 'state/T'
    gas_constant = 8.0
    growth_rate = 'state/R_dot'
  []
[]
