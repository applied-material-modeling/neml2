[Tensors]
  [A]
    type = FillSR2
    values = '1 2 3 4 5 6'
  []
  [B]
    type = FillSR2
    values = '-1 -4 7 -1 9 1'
  []
  [C]
    type = FillSR2
    values = '-6.5 -30 58.5 -5.292893219 75.20710678 11.70710678'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'state/A state/B'
    input_SR2_values = 'A B'
    output_SR2_names = 'state/C'
    output_SR2_values = 'C'
  []
[]

[Models]
  [model]
    type = SR2LinearCombination
    from_var = 'state/A state/B'
    to_var = 'state/C'
    coefficients = '0.5 8'
    constant_coefficient = '1.0'
  []
[]
