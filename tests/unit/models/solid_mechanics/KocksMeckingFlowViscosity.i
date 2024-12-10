[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'p'
    input_scalar_names = 'forces/T'
    input_scalar_values = '1000'
    output_scalar_names = 'parameters/p'
    output_scalar_values = 'p_correct'
    check_second_derivatives = true
  []
[]

[Models]
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
[]

[Tensors]
  [mu]
    type = LinspaceScalar
    start = 50000
    end = 100000
    nstep = 5
  []
  [A]
    type = LinspaceScalar
    start = -3.5
    end = -5.5
    nstep = 5
  []
  [B]
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
