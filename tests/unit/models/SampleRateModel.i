[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'state/foo state/bar forces/temperature'
    input_scalar_values = '1 2 100'
    input_symr2_names = 'state/baz'
    input_symr2_values = '0.5'
    output_scalar_names = 'state/foo_rate state/bar_rate'
    output_scalar_values = '301.5 -89.02'
    output_symr2_names = 'state/baz_rate'
    output_symr2_values = '145.5'
  []
[]

[Models]
  [model]
    type = SampleRateModel
  []
[]
