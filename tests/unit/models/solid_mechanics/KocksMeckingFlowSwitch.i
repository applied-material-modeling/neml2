[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'forces/g state/internal/ri_rate state/internal/rd_rate'
    input_scalar_values = 'g 0.5 0.75'
    output_scalar_names = 'state/internal/gamma_rate'
    output_scalar_values = 'fr_correct'
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
