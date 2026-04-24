[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'a b'
    input_Scalar_values = '3.1 2.5'
    output_Scalar_names = 'complementarity'
    output_Scalar_values = '-9.58246155'
  []
[]

[Models]
  [model]
    type = FBComplementarity
    a_inequality = 'LE'
    b_inequality = 'LE'
  []
[]
