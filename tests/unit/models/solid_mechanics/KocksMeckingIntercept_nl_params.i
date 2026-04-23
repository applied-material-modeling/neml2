[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'A B C'
    input_Scalar_values = 'A_in B_in C_in'
    output_Scalar_names = 'intercept'
    output_Scalar_values = 'p_correct'
    check_second_derivatives = true
  []
[]

[Models]
  [p]
    type = KocksMeckingIntercept
    A = 'A'
    B = 'B'
    C = 'C'
  []
  [model]
    type = ComposedModel
    models = 'p'
  []
[]

[Tensors]
  [A_in]
    type = LinspaceScalar
    start = -2.0
    end = -3.0
    nstep = 5
  []
  [B_in]
    type = LinspaceScalar
    start = -4.0
    end = -7.0
    nstep = 5
  []
  [C_in]
    type = LinspaceScalar
    start = -5.0
    end = -8.0
    nstep = 5
  []
  [p_correct]
    type = Scalar
    values = "0.5 0.44444444 0.4 0.36363636 0.33333333"
    batch_shape = '(5)'
  []
[]
