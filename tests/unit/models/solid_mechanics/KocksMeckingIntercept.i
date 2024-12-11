[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'p'
    batch_shape = '(5)'
    output_scalar_names = 'parameters/p'
    output_scalar_values = 'p_correct'
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
[]

[Tensors]
  [A]
    type = LinspaceScalar
    start = -2.0
    end = -3.0
    nstep = 5
  []
  [B]
    type = LinspaceScalar
    start = -4.0
    end = -7.0
    nstep = 5
  []
  [C]
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
