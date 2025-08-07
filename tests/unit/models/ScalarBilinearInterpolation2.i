# Batched ordinate:
#
# abscissa1: (     3;)
# abscissa2: (     2;)
#  ordinate: (2; 3,2;)
# -------------------------
# argument1: (      ;)
# argument2: (      ;)
# -------------------------
#    output: (2;    ;)

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_Scalar_names = 'forces/T forces/eps'
    input_Scalar_values = '0.5 0.5'
    output_Scalar_names = 'parameters/E'
    output_Scalar_values = 'E_correct'
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
    intermediate_dimension = 1
  []
  [eps]
    type = Scalar
    values = '0 2'
    batch_shape = '(2)'
    intermediate_dimension = 1
  []
  [S]
    type = Scalar
    values = "1 2 3 4 5 6
              2 3 4 5 6 7"
    batch_shape = '(2,3,2)'
    intermediate_dimension = 2
  []
  [E_correct]
    type = Scalar
    values = '2.25 3.25'
    batch_shape = '(2)'
  []
[]
