[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'forces/T forces/effective_strain_rate state/mu'
    input_Scalar_values = 'T 1.1 75000'
    output_Scalar_names = 'forces/g'
    output_Scalar_values = 'g_correct'
  []
[]

[Models]
  [model0]
    type = KocksMeckingActivationEnergy
    eps0 = 1e10
    k = 1.38064e-20
    b = 2.019e-7
    shear_modulus = 'state/mu'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]

[Tensors]
  [T]
    type = LinspaceScalar
    start = 500
    end = 1000
    nstep = 5
  []
  [g_correct]
    type = Scalar
    values = "0.25644517 0.32055647 0.38466776 0.44877906 0.51289035"
    batch_shape = '(5)'
  []
[]
