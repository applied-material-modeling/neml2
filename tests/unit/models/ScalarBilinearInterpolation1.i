# The unbatched case:
#
# abscissa1: (  3;)
# abscissa2: (  2;)
#  ordinate: (3,2;)
# -------------------------
# argument1: (  ;)
# argument2: (  ;)
# -------------------------
#    output: (  ;)

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_Scalar_names = 'forces/T forces/eps'
    input_Scalar_values = '0.5 0.5'
    output_Scalar_names = 'parameters/E'
    output_Scalar_values = '2.25'
    check_second_derivatives = true
    check_AD_parameter_derivatives = false
  []
[]

[Models]
  [E]
    type = ScalarBilinearInterpolation
    argument1 = 'forces/T'
    argument2 = 'forces/eps'
    abscissa1 = 'T'
    abscissa2 = 'eps'
    ordinate = 'S'
  []
[]

[Tensors]
  [T]
    type = Scalar
    values = '0 1 2'
    batch_shape = '(3)'
  []
  [eps]
    type = Scalar
    values = '0 2'
    batch_shape = '(2)'
  []
  [S]
    type = Scalar
    values = '1 2 3 4 5 6'
    batch_shape = '(3,2)'
  []
[]
