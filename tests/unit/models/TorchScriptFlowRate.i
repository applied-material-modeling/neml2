[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 's T'
    input_Scalar_values = '1e6 300'
    output_Scalar_names = 'ep_rate'
    output_Scalar_values = '6.55616e-09'
  []
[]

[Models]
  [model]
    type = TorchScriptFlowRate
    torch_script = 'TorchScriptFlowRate.pt'
    von_mises_stress = 's'
    temperature = 'T'
    equivalent_plastic_strain_rate = 'ep_rate'
  []
[]
