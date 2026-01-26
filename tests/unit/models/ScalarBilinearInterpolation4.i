# Everything batched:
#
# abscissa1: (2;   3;)
# abscissa2: (2;   2;)
#  ordinate: (2; 3,2;)
# -------------------------
# argument1: (2;    ;)
# argument2: (2;    ;)
# -------------------------
#    output: (2;    ;)

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
    values = "0 1 2
              10 20 30"
    batch_shape = '(2,3)'
    intermediate_dimension = 1
  []
  [eps_vals]
    type = Scalar
    values = "0 2
              1 3"
    batch_shape = '(2,2)'
    intermediate_dimension = 1
  []
  [S]
    type = Scalar
    values = "1 2 3 4 5 6
              -1 -2 -3 -4 -5 -6"
    batch_shape = '(2,3,2)'
    intermediate_dimension = 2
  []
  [T]
    type = Scalar
    values = '0.5 21'
    batch_shape = '(2)'
  []
  [eps]
    type = Scalar
    values = '0.5 1.6'
    batch_shape = '(2)'
  []
  [E_correct]
    type = Scalar
    values = '2.25 -3.5'
    batch_shape = '(2)'
  []
[]
