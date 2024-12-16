[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'params/n params/tau state/internal/k'
    input_Scalar_values = '2.0 75.0 125.0'
    output_Scalar_names = 'state/internal/k_recovery_rate'
    output_Scalar_values = '-2.7777777778'
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
