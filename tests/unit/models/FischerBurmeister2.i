[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/a state/b'
      input_Scalar_values = '3.1 2.5'
      output_Scalar_names = 'state/fb'
      output_Scalar_values = '-4.58246155'
    []
  []
  
  [Models]
    [model]
      type = FischerBurmeister
      first_var = 'state/a'
      second_var = 'state/b'
      fischer_burmeister = 'state/fb'
      first_inequality = 'LE'
    []
  []