[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'A'
    input_Scalar_values = 'A'
    output_Scalar_names = 'foo'
    output_Scalar_values = 'foo'
    input_with_intrsc_intmd_dims = 'A'
    input_intrsc_intmd_dims = '1'
  []
[]

[Tensors]
  [A]
    type = Scalar
    values = '1.0 2.0 3.0'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [foo]
    type = Scalar
    values = '6.0'
  []
[]

[Models]
  [comb]
      type = ScalarLinearCombination
      from_var = 'A'
      to_var = 'U'
      coefficients = '1'
  []
  [sum]
    type = ScalarIntermediateSum
    from = 'U'
    to = 'foo'
  []
  [model]
    type = ComposedModel
    models = 'comb sum'
  []
[]
