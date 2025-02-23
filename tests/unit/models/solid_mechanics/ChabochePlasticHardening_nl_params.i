[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/gamma_rate state/C state/g state/A state/a'
    input_Scalar_values = '0.01 1000.0 10.0 1.0e-6 2.1'
    input_SR2_names = 'state/internal/NM state/internal/X'
    input_SR2_values = 'NM X'
    output_SR2_names = 'state/internal/X_rate'
    output_SR2_values = 'X_rate'
    value_abs_tol = 1.0e-4
  []
[]

[Tensors]
  [NM]
    type = FillSR2
    values = '-0.3482 0.3482 0 0.087045 0.087045 0.78333'
  []
  [X]
    type = FillSR2
    values = '-10 15 5 -7 15 20'
  []
  [X_rate]
    type = FillSR2
    values = '-1.3207 0.8204 -0.5003 1.2807 -0.9206 3.2210'
  []
[]

[Models]
  [model0]
    type = ChabochePlasticHardening
    C = 'state/C'
    g = 'state/g'
    A = 'state/A'
    a = 'state/a'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
