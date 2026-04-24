[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'flow_rate'
    input_Scalar_values = '0.01'
    input_SR2_names = 'flow_direction back_stress'
    input_SR2_values = 'NM X'
    output_SR2_names = 'back_stress_rate'
    output_SR2_values = 'X_rate'
    value_abs_tol = 1e-4
  []
[]

[Tensors]
  [NM]
    type = FillSR2
    values = '-0.3482 0.3482 0 0.087045 0.087045 0.78333'
  []
  [X]
    type = FillSR2
    values = '-10 15 5 -7 15 20'
  []
  [X_rate]
    type = FillSR2
    values = '-1.3213 0.8213 -0.5000 1.2803 -0.9197 3.2222'
  []
[]

[Models]
  [model]
    type = FredrickArmstrongPlasticHardening
    C = 1000
    g = 10
  []
[]
