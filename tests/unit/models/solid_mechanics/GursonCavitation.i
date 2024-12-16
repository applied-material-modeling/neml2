[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/f'
    input_Scalar_values = '0.1'
    input_SR2_names = 'state/internal/Ep_rate'
    input_SR2_values = 'Epr'
    output_Scalar_names = 'state/internal/f_rate'
    output_Scalar_values = '0.1125'
  []
[]

[Tensors]
  [Epr]
    type = FillSR2
    values = '0.1 0.05 -0.025 0.15 -0.2 0.5'
  []
[]

[Models]
  [model]
    type = GursonCavitation
  []
[]
