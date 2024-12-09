[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/s forces/T state/G state/C'
    input_Scalar_values = '1e6 300 0.1 0.2'
    output_Scalar_names = 'state/ep_rate state/G_rate state/C_rate'
    output_Scalar_values = '6.55616e-09 0.005 0.002'
  []
[]

[Models]
  [model]
    type = TorchScriptFlowRate
    torch_script = 'TorchScriptFlowRate.pt'
  []
[]
