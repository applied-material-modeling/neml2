[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/mu state/C'
    input_Scalar_values = 'mu_in C_in'
    output_Scalar_names = 'parameters/p'
    output_Scalar_values = 'p_correct'
    check_second_derivatives = true
  []
[]

[Models]
  [p]
    type = KocksMeckingYieldStress
    C = 'state/C'
    shear_modulus = 'state/mu'
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
  [C_in]
    type = LinspaceScalar
    start = -3.5
    end = -5.5
    nstep = 5
  []
  [p_correct]
    type = Scalar
    values = "1509.86917112 1144.72743055 833.17474037 589.57036242 408.67714385"
    batch_shape = '(5)'
  []
[]
