# neml2
# Past the failure fraction the map saturates at ultimate_fraction instead of
# extrapolating. f = 0.95 is well beyond the point where the unsaturated
# accelerated branch would have driven f* through one
# (f_c + (1 - f_c) / kappa = 0.8061224490), so the value pins the cap and the
# driver's JVP check pins the tangents: zero in f, f_c and f_F, exactly one in
# f_u. Promoting all three parameters is what makes the last claim testable.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'cavity_fraction coalescence_fraction failure_fraction ultimate_fraction'
    input_Scalar_values = 'f f_c f_F f_u'
    output_Scalar_names = 'effective_cavity_fraction'
    output_Scalar_values = 'f_star'
    value_abs_tol = 1e-9
  []
[]

[Tensors]
  [f]
    type = Python
    expr = 'Scalar(0.95)'
  []
  [f_c]
    type = Python
    expr = 'Scalar(0.5)'
  []
  [f_F]
    type = Python
    expr = 'Scalar(0.8)'
  []
  [f_u]
    type = Python
    expr = 'Scalar(0.99)'
  []
  [f_star]
    type = Python
    expr = 'Scalar(0.99)'
  []
[]

[Models]
  [model]
    type = EffectiveCavityFraction
    coalescence_fraction = 'coalescence_fraction'
    failure_fraction = 'failure_fraction'
    ultimate_fraction = 'ultimate_fraction'
  []
[]
