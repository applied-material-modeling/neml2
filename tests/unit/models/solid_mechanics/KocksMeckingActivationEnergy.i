[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'temperature strain_rate'
    input_Scalar_values = 'T 1.1'
    output_Scalar_names = 'activation_energy'
    output_Scalar_values = 'g_correct'
    check_AD_parameter_derivatives = false
  []
[]

[Models]
  [model]
    type = KocksMeckingActivationEnergy
    eps0 = 1e10
    k = 1.38064e-20
    b = 2.019e-7
    shear_modulus = 75000.0
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
