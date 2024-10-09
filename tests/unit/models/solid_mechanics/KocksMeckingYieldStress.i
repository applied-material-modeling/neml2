[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'p'
    output_scalar_names = 'parameters/p'
    output_scalar_values = 'p_correct'
  []
[]

[Models]
  [p]
    type = KocksMeckingYieldStress
    C = 'C'
    shear_modulus = 'mu'
  []
[]

[Tensors]
  [mu]
    type = LinspaceScalar
    start = 50000
    end = 100000
    nstep = 5
  []
  [C]
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
