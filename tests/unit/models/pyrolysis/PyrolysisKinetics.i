[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/T state/f'
      input_Scalar_values = 'T f'
      output_Scalar_names = 'state/out'
      output_Scalar_values = 'out'
      check_AD_parameter_derivatives = false
  []
[]

[Tensors]
    [T]
        type = Scalar
        values = "100 52300 -7"
        batch_shape = '(3)'
    []
    [f]
        type = Scalar
        values = "0.1 123 44"
        batch_shape = '(3)'
    []
    [out]
        type = Scalar
        values = "2.0999218e-4 0.2583 0.092449"
        batch_shape = '(3)'
    []
[]

[Models]
    [model]
        type = PyrolysisKinetics
        kinetic_constant = 2.1e-3
        activation_energy = 6.7e-5
        ideal_gas_constant = 1.8e-2
        temperature = 'state/T'
        reaction = 'state/f'
        out = 'state/out'
    []
[]