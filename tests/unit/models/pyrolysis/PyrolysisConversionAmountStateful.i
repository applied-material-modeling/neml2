[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/ms state/ms0 state/mb0'
      input_Scalar_values = 'ms 0.2 0.4'
      output_Scalar_names = 'state/alpha'
      output_Scalar_values = 'alpha'
      check_AD_parameter_derivatives = false
  []
[]

[Tensors]
    [ms]
        type = Scalar
        values = "0.03 0.75 0.22"
        batch_shape = '(3)'
    []
    [alpha]
        type = Scalar
        values = "4.75 -1.25 3.166666"
        batch_shape = '(3)'
    []
[]

[Models]
    [model]
        type = PyrolysisConversionAmountStateful
        initial_mass_solid = 'state/ms0'
        initial_mass_binder = 'state/mb0'
        reaction_yield = 0.7

        mass_solid = 'state/ms'
        reaction_amount = 'state/alpha'
    []
[]