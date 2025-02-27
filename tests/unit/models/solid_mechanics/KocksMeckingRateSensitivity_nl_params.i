[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'forces/T state/mu state/A'
    input_Scalar_values = '1000 mu_in A_in'
    output_Scalar_names = 'parameters/p'
    output_Scalar_values = 'p_correct'
    check_second_derivatives = true
  []
[]

[Models]
  [p]
    type = KocksMeckingRateSensitivity
    shear_modulus = 'state/mu'
    A = 'state/A'
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
