# Translated from tests/unit/models/solid_mechanics/crystal_plasticity/SumSlipRates.i.
# The model reduces over the slip axis (sub-batch trailing dim) — per-slip Scalar
# in, single Scalar out. Sum of absolute values of 12 slip rates equals 1.91.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'slip_rates'
    input_Scalar_values = 'rates'
    output_Scalar_names = 'sum_slip_rates'
    output_Scalar_values = '1.91'
  []
[]

[Tensors]
  [rates]
    type = Python
    expr = 'Scalar(torch.tensor([-0.2, -0.15, -0.1, -0.05, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35]), sub_batch_ndim=1)'
  []
[]

[Models]
  [model]
    type = SumSlipRates
  []
[]
