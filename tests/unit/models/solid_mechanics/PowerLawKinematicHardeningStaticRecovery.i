[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/tau state/n'
    input_Scalar_values = '120.0 2.0'
    input_SR2_names = 'state/internal/X'
    input_SR2_values = 'X'
    output_SR2_names = 'state/internal/X_recovery_rate'
    output_SR2_values = 'X_rate'
    value_abs_tol = 1.0e-4
  []
[]

[Tensors]
  [X]
    type = FillSR2
    values = '-10 15 5 -7 15 20'
  []
  [X_rate]
    type = FillSR2
    values = '0.02861583 -0.04292375 -0.01430792 0.02003108 -0.04292375 -0.05723166'
  []
[]

[Models]
  [model0]
    type = PowerLawKinematicHardeningStaticRecovery
    tau = 'state/tau'
    n = 'state/n'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
