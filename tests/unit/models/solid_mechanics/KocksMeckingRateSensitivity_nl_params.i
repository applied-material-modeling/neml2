[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'forces/T params/mu params/A'
    input_scalar_values = '1000 mu_in A_in'
    output_scalar_names = 'parameters/p'
    output_scalar_values = 'p_correct'
    check_second_derivatives = true
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
  [p]
    type = KocksMeckingRateSensitivity
    shear_modulus = 'mu'
    A = 'A'
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
  [p_correct]
    type = Scalar
    values = "8.51589828 9.31426374 9.93521466 10.43197539 10.83841599"
    batch_shape = '(5)'
  []
[]
