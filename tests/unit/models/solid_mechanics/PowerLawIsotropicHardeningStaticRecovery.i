[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'params/n params/tau state/internal/k'
    input_scalar_values = '2.0 75.0 125.0'
    output_scalar_names = 'state/internal/k_recovery_rate'
    output_scalar_values = '-2.7777777778'
  []
[]

[Models]
  [n]
    type = ScalarInputParameter
    from = 'params/n'
  []
  [tau]
    type = ScalarInputParameter
    from = 'params/tau'
  []
  [model0]
    type = PowerLawIsotropicHardeningStaticRecovery
    tau = 'tau'
    n = 'n'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
