# Translated from
# tests/unit/models/solid_mechanics/plasticity/PowerLawIsotropicHardeningStaticRecovery.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'n tau isotropic_hardening'
    input_Scalar_values = '2.0 75.0 125.0'
    output_Scalar_names = 'isotropic_hardening_rate'
    output_Scalar_values = '-2.7777777778'
  []
[]

[Models]
  [model]
    type = PowerLawIsotropicHardeningStaticRecovery
    tau = 'tau'
    n = 'n'
  []
[]
