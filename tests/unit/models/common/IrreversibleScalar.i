# neml2
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'x_trial damage~1'
    input_Scalar_values = '0.5 0.3'
    output_Scalar_names = 'damage'
    # Trial > history -> output advances to trial value.
    output_Scalar_values = '0.5'
    derivative_abs_tol = 1e-6
  []
[]

[Models]
  [model]
    type = IrreversibleScalar
    from = 'x_trial'
    to = 'damage'
  []
[]
