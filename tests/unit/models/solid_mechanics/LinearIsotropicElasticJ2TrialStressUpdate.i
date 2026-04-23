[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'elastic_trial_stress equivalent_plastic_strain equivalent_plastic_strain~1'
    input_Scalar_values = '1.5 0.1 0.05'
    output_Scalar_names = 'updated_trial_stress'
    output_Scalar_values = '1.38461538462'
    derivative_abs_tol = 1e-6
  []
[]

[Models]
  [model]
    type = LinearIsotropicElasticJ2TrialStressUpdate
    coefficients = '2 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
