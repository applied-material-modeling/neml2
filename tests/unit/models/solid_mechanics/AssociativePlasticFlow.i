[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'state/internal/gamma_rate'
    input_scalar_values = '0.0015'
    input_symr2_names = 'state/internal/NM'
    input_symr2_values = 'NX'
    output_symr2_names = 'state/internal/Ep_rate'
    output_symr2_values = 'Ep_rate'
  []
[]

[Tensors]
  [NX]
    type = FillSR2
    values = '-0.3482 0.3482 0 0.087045 0.087045 0.78333'
  []
  [Ep_rate]
    type = FillSR2
    values = '-0.0005223 0.0005223 0 0.0001305675 0.0001305675 0.001174995'
  []
[]

[Models]
  [model]
    type = AssociativePlasticFlow
  []
[]
