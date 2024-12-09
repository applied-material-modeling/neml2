[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'forces/s state/ep old_state/ep'
    input_scalar_values = '1.5 0.1 0.05'
    output_scalar_names = 'state/s'
    output_scalar_values = '1.38461538462'
    derivative_abs_tol = 1e-6
  []
[]

[Models]
  [model]
    type = LinearIsotropicElasticJ2TrialStressUpdate
    youngs_modulus = 2
    poisson_ratio = 0.3
  []
[]
