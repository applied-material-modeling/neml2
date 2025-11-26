# Batched argument case:
#
# abscissa1: (;     3;)
# abscissa2: (;     2;)
#  ordinate: (;   3,2;)
# -------------------------
# argument1: (4;     ;)
# argument2: (1;     ;)
# -------------------------
#    output: (4;     ;)

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_Scalar_names = 'forces/T forces/eps'
    input_Scalar_values = 'T eps'
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
    abscissa1 = 'T_vals'
    abscissa2 = 'eps_vals'
    ordinate = 'S'
  []
[]

[Tensors]
  [T_vals]
    type = Scalar
    values = '0 1 2'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [eps_vals]
    type = Scalar
    values = '0 2'
    batch_shape = '(2)'
    intermediate_dimension = 1
  []
  [S]
    type = Scalar
    values = '1 2 3 4 5 6'
    batch_shape = '(3,2)'
    intermediate_dimension = 2
  []
  [T]
    type = Scalar
    values = '0.5 0.75 1.25 1.75'
    batch_shape = '(4)'
  []
  [eps]
    type = Scalar
    values = '0.5'
    batch_shape = '(1)'
  []
  [E_correct]
    type = Scalar
    values = '2.25 2.75 3.75 4.75'
    batch_shape = '(4)'
  []
[]
