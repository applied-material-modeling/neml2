# Cover the optional-isotropic_hardening branch: when the HIT option is
# omitted, the model should not declare an isotropic_hardening input and
# evaluate f = sqrt(2/3) * (effective_stress - sy)
#                = sqrt(2/3) * (1500 - 1000) = 408.2482904638630.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'effective_stress'
    input_Scalar_values = 'es'
    output_Scalar_names = 'yield_function'
    output_Scalar_values = 'fy'
  []
[]

[Tensors]
  [es]
    type = Python
    expr = 'Scalar(1500.0)'
  []
  [fy]
    type = Python
    expr = 'Scalar(408.24829046386304)'
  []
[]

[Models]
  [model]
    type = YieldFunction
    yield_stress = 1000.0
  []
[]
