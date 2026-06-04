# Translated from tests/unit/models/solid_mechanics/crystal_plasticity/PerSlipForestDislocationEvolution.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'dislocation_density slip_rates'
    input_Scalar_values = 'rho gamma_dot'
    output_Scalar_names = 'dislocation_density_rate'
    output_Scalar_values = 'rho_dot'
    derivative_rel_tol = 0
    derivative_abs_tol = 5e-6
  []
[]

[Tensors]
  [rho]
    type = Python
    expr = 'Scalar([1.0, 4.0, 9.0, 16.0]).sub_batch.retag(1)'
  []
  [gamma_dot]
    type = Python
    expr = 'Scalar([0.1, -0.2, 0.3, -0.4]).sub_batch.retag(1)'
  []
  [rho_dot]
    type = Python
    expr = 'Scalar([0.15, 0.4, 0.45, 0.0]).sub_batch.retag(1)'
  []
[]

[Models]
  [model]
    type = PerSlipForestDislocationEvolution
    k1 = 2.0
    k2 = 0.5
  []
[]
