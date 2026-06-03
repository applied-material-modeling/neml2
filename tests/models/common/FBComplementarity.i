# Translated from tests/unit/models/common/FischerBurmeister1.i.
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
    type = FBComplementarity
    a_inequality = 'GE'
    b_inequality = 'GE'
  []
[]
