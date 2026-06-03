[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'phi'
    input_Scalar_values = 'phi'
    output_Scalar_names = 'K'
    output_Scalar_values = 'K'
    derivative_abs_tol = 1e-7
  []
[]

[Tensors]
  [phi]
    type = Python
    expr = 'Scalar([0.01, 0.65, 0.98])'
  []
  [phi0]
    type = Python
    expr = 'Scalar([0.71, 0.35, 1.0])'
  []
  [a]
    type = Python
    expr = 'Scalar([0.0, 0.4, 1.8])'
  []
  [K]
    type = Python
    expr = 'Scalar([3.0, 2.66076131, 3.109967539])'
  []
[]

[Models]
  [model]
    type = ExponentialLawPermeability
    reference_permeability = 3
    reference_porosity = 'phi0'
    scale = 'a'
    porosity = 'phi'
    permeability = 'K'
  []
[]
