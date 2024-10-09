[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_skewr2_names = 'state/foo_rate'
    input_skewr2_values = 'w'
    input_rot_names = 'state/foo old_state/foo'
    input_rot_values = 'foo old_foo'
    input_scalar_names = 'forces/t old_forces/t'
    input_scalar_values = '1.3 1.1'
    output_rot_names = 'residual/foo'
    output_rot_values = 'res'
    value_rel_tol = 1e-4
  []
[]

[Models]
  [model]
    type = WR2ImplicitExponentialTimeIntegration
    variable = 'state/foo'
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
