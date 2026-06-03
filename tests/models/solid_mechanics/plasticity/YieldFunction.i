# No C++ ModelUnitTest fixture (YieldFunction is exercised inside a composed
# flow in C++). Values hand-computed: f = sqrt(2/3) * (effective_stress - sy -
# isotropic_hardening) = sqrt(2/3) * (1500 - 1000 - 100).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'effective_stress isotropic_hardening'
    input_Scalar_values = 'es h'
    output_Scalar_names = 'yield_function'
    output_Scalar_values = 'fy'
  []
[]

[Tensors]
  [es]
    type = Python
    expr = 'Scalar(1500.0)'
  []
  [h]
    type = Python
    expr = 'Scalar(100.0)'
  []
  [fy]
    type = Python
    expr = 'Scalar(326.5986323710904)'
  []
[]

[Models]
  [model]
    type = YieldFunction
    yield_stress = 1000.0
    isotropic_hardening = 'isotropic_hardening'
  []
[]
