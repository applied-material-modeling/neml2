[Tensors]
  [fp]
    type = Scalar
    values = '-2 -1 0.5 1 2'
    batch_shape = '(5)'
  []
  [gamma_rate]
    type = Scalar
    values = '5 4 3 2 1'
    batch_shape = '(5)'
  []
  [rp]
    type = Scalar
    values = '1.61483519 0.87689437 -0.54138127 -1.23606798 -3.23606798'
    batch_shape = '(5)'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'state/internal/fp state/internal/gamma_rate'
    input_scalar_values = 'fp gamma_rate'
    output_scalar_names = 'residual/internal/gamma_rate'
    output_scalar_values = 'rp'
  []
[]

[Models]
  [model]
    type = RateIndependentPlasticFlowConstraint
  []
[]
