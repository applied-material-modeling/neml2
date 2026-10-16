# neml2
# All three coalescence parameters promoted to runtime inputs, on the
# accelerated branch where every one of them is load-bearing. The driver
# finite-differences the JVP, so this is what catches a parameter that
# changes f* but contributes nothing to the pushforward.
#
# kappa = (0.99 - 0.5) / (0.8 - 0.5) = 1.6333333333
# f*    = 0.5 + (0.99 - 0.5) * (0.6 - 0.5) / (0.8 - 0.5) = 0.6633333333
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
    expr = 'Scalar(0.6)'
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
    expr = 'Scalar(0.6633333333333333)'
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
