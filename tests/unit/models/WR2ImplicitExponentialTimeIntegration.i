[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_WR2_names = 'foo_rate'
    input_WR2_values = 'w'
    input_Rot_names = 'foo foo~1'
    input_Rot_values = 'foo old_foo'
    input_Scalar_names = 't t~1'
    input_Scalar_values = '1.3 1.1'
    output_Rot_names = 'foo_residual'
    output_Rot_values = 'res'
    value_rel_tol = 1e-4
  []
[]

[Models]
  [model]
    type = WR2ImplicitExponentialTimeIntegration
    variable = 'foo'
    time = 't'
  []
[]

[Tensors]
  [res]
    type = FillRot
    values = '-0.032903 -0.005864  0.006609'
  []
  [foo]
    type = FillRot
    values = '0.00499066 -0.0249533 0.03493462'
  []
  [old_foo]
    type = FillRot
    values = '0.03739906 -0.01994617  0.02991925'
  []
  [w]
    type = FillWR2
    values = '0.01 0.02 -0.03'
  []
[]
