[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'forces/T params/A params/B params/mu'
    input_scalar_values = '1000 A_in B_in mu_in'
    output_scalar_names = 'parameters/p'
    output_scalar_values = 'p_correct'
    check_second_derivatives = true
    derivative_abs_tol = 0.01
    second_derivative_abs_tol = 1e-3
  []
[]

[Models]
  [mu]
    type = ScalarInputParameter
    from = 'params/mu'
  []
  [A]
    type = ScalarInputParameter
    from = 'params/A'
  []
  [B]
    type = ScalarInputParameter
    from = 'params/B'
  []
  [p]
    type = KocksMeckingFlowViscosity
    shear_modulus = 'mu'
    A = 'A'
    B = 'B'
    eps0 = 1e10
    k = 1.38064e-20
    b = 2.019e-7
    temperature = 'forces/T'
  []
  [model]
    type = ComposedModel
    models = 'p'
  []
[]

[Tensors]
  [mu_in]
    type = LinspaceScalar
    start = 50000
    end = 100000
    nstep = 5
  []
  [A_in]
    type = LinspaceScalar
    start = -3.5
    end = -5.5
    nstep = 5
  []
  [B_in]
    type = LinspaceScalar
    start = -1.5
    end = -3.0
    nstep = 5
  []
  [p_correct]
    type = Scalar
    values = "746.88551856 809.01337039 778.71385139 697.25850376 594.93909239"
    batch_shape = '(5)'
  []
[]
