# SmoothCap with a sharp exponent (p = 200). Evaluated directly,
# x^-p + xmax^-p underflows to 0 here (50^-200 < the smallest float64) and the
# result overflows to inf; the model's ratio form stays finite. Reference values
# from 30-digit arithmetic, y = (x^-200 + 50^-200)^(-1/200):
#   x = 10    -> 10
#   x = 50    -> 49.827013141393392  (2^(-1/200) * 50)
#   x = 100   -> 50
#   x = 1000  -> 50
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'x'
    input_Scalar_values = 'x'
    output_Scalar_names = 'y'
    output_Scalar_values = 'y'
  []
[]

[Tensors]
  [x]
    type = Python
    expr = "Scalar(torch.tensor([10.0, 50.0, 100.0, 1000.0], dtype=torch.float64))"
  []
  [y]
    type = Python
    expr = "Scalar(torch.tensor([10.0, 49.827013141393392, 50.0, 50.0], dtype=torch.float64))"
  []
[]

[Models]
  [model]
    type = SmoothCap
    argument = 'x'
    value = 'y'
    maximum = 50.0
    exponent = 200.0
  []
[]
