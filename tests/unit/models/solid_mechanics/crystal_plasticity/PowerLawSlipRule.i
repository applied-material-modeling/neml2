[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/resolved_shears state/internal/slip_strengths'
    input_Scalar_values = 'tau tau_bar'
    output_Scalar_names = 'state/internal/slip_rates'
    output_Scalar_values = 'rates'
    check_AD_parameter_derivatives = false
    derivative_rel_tol = 0
    derivative_abs_tol = 5e-6
    second_derivative_rel_tol = 0
    second_derivative_abs_tol = 5e-6
  []
[]

[Tensors]
  [tau]
    type = LinspaceScalar
    start = -100
    end = 200
    nstep = 12
    group = 'intermediate'
  []
  [tau_bar]
    type = LinspaceScalar
    start = 50
    end = 250
    nstep = 12
    group = 'intermediate'
  []
  [rates]
    type = Scalar
    values = '-3.4297e-02 -1.3898e-03 -3.7875e-05 -1.3357e-07 1.7191e-09 9.9957e-07 9.3434e-06 3.3176e-05 7.6855e-05 1.4079e-04 2.2299e-04 3.2045e-04'
    batch_shape = '(12)'
    intermediate_dimension = 1
  []
[]

[Models]
  [model]
    type = PowerLawSlipRule
    n = 5.1
    gamma0 = 1e-3
  []
[]
