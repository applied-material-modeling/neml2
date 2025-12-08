[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/resolved_shears state/internal/slip_strengths forces/T'
    input_Scalar_values = 'tau tau_bar T'
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
  [T]
    type = LinspaceScalar
    start = 350
    end = 550
    nstep = 3
  []
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
  [gamma0_x]
    type = Scalar
    values = '300 400 500 600'
    batch_shape = '(1,4)'
    intermediate_dimension = 2
  []
  [gamma0_y]
    type = Scalar
    values = '1e-3 1.5e-3 1.8e-3 2.1e-3'
    batch_shape = '(1,4)'
    intermediate_dimension = 2
  []
  [rates]
    type = Scalar
    values = "-0.0428709 -0.00173723 -4.73442e-05 -1.66957e-07 2.14886e-09 1.24947e-06 1.16793e-05 4.14699e-05 9.60694e-05 0.000175985 0.000278739 0.000400561
              -0.0565896 -0.00229314 -6.24943e-05 -2.20383e-07 2.83649e-09 1.64929e-06 1.54166e-05 5.47403e-05 0.000126812 0.0002323 0.000367936 0.000528741
              -0.0668787 -0.00271007 -7.38569e-05 -2.60453e-07 3.35222e-09 1.94917e-06 1.82197e-05 6.46931e-05 0.000149868 0.000274537 0.000434834 0.000624876"
    batch_shape = '(3,12)'
    intermediate_dimension = 1
  []
[]

[Models]
  [gamma0_per_slip]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = 'gamma0_x'
    ordinate = 'gamma0_y'
  []
  [model0]
    type = PowerLawSlipRule
    n = 5.1
    gamma0 = 'gamma0_per_slip'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
