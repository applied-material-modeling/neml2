# Translated from
# tests/unit/models/solid_mechanics/plasticity/PowerLawKinematicHardeningStaticRecovery.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'tau n'
    input_Scalar_values = '120.0 2.0'
    input_SR2_names = 'back_stress'
    input_SR2_values = 'X'
    output_SR2_names = 'back_stress_rate'
    output_SR2_values = 'X_rate'
    value_abs_tol = 1.0e-4
  []
[]

[Tensors]
  [X]
    type = Python
    expr = 'SR2.fill(-10, 15, 5, -7, 15, 20)'
  []
  [X_rate]
    type = Python
    expr = 'SR2.fill(0.02861583, -0.04292375, -0.01430792, 0.02003108, -0.04292375, -0.05723166)'
  []
[]

[Models]
  [model]
    type = PowerLawKinematicHardeningStaticRecovery
    tau = 'tau'
    n = 'n'
  []
[]
