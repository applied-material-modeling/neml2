[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'a b'
    input_Scalar_values = '3.1 2.5'
    output_Scalar_names = 'complementarity'
    output_Scalar_values = '1.61753845'
  []
[]

[Models]
  [model]
    type = FischerBurmeister
  []
[]
