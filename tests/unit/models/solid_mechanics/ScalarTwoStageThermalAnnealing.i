[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'forces/T state/internal/k_rate_unmodified state/internal/k'
    input_scalar_values = 'temperature_in 20.0 100.0'
    output_scalar_names = 'state/internal/k_rate'
    output_scalar_values = 'correct_values'
    check_second_derivatives = false
  []
[]

[Models]
  [model]
    type = ScalarTwoStageThermalAnnealing
    base_rate = 'state/internal/k_rate_unmodified'
    base = 'state/internal/k'
    modified_rate = 'state/internal/k_rate'
    temperature = 'forces/T'

    T1 = 1000.0
    T2 = 1200.0

    tau = 20.0
  []
[]

[Tensors]
  [temperature_in]
    type = Scalar
    values = "800.0 999 1001 1100 1199 1201 1250"
    batch_shape = '(7)'
  []

  [correct_values]
    type = Scalar
    values = "20.0 20.0 0.0 0.0 0.0 -5.0 -5.0"
    batch_shape = '(7)'
  []
[]
