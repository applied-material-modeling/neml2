[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'p'
    input_Scalar_names = 'T'
    input_Scalar_values = '1000'
    output_Scalar_names = 'p'
    output_Scalar_values = 'p_correct'
    check_second_derivatives = true
  []
[]

[Models]
  [p]
    type = ArrheniusParameter
    temperature = 'T'
    reference_value = 'p0'
    activation_energy = 'Q'
    ideal_gas_constant = 8.314
  []
[]

[Tensors]
  [p0]
    type = LinspaceScalar
    start = 1
    end = 10
    nstep = 5
  []
  [Q]
    type = LinspaceScalar
    start = 1e3
    end = 2e4
    nstep = 5
  []
  [p_correct]
    type = Scalar
    values = "0.8866729736328125
              1.6275087594985962
              1.5555329322814941
              1.2379260063171387
              0.9021309018135071"
    batch_shape = '(5)'
  []
[]
