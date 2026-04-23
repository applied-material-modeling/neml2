[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'activation_energy rate_independent_flow_rate rate_dependent_flow_rate'
    input_Scalar_values = 'g 0.5 0.75'
    output_Scalar_names = 'flow_rate'
    output_Scalar_values = 'fr_correct'
    check_AD_parameter_derivatives = false
  []
[]

[Models]
  [model]
    type = KocksMeckingFlowSwitch
    g0 = 0.5
    sharpness = 2.1
  []
[]

[Tensors]
  [g]
    type = LinspaceScalar
    start = 0.1
    end = 0.9
    nstep = 5
  []
  [fr_correct]
    type = Scalar
    values = "0.53927387 0.5753837 0.625 0.6746163 0.71072613"
    batch_shape = '(5)'
  []
[]
