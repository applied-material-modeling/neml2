[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/alpha'
      input_Scalar_values = '0.053456'
      output_Scalar_names = 'state/out'
      output_Scalar_values = '0.6134410835'
    []
  []
  
  [Models]
    [model]
      type = EffectiveSaturation
      residual_saturation = 0.114
      molar_volume = 12.3
      saturation = 'state/alpha'
      effective_saturation = 'state/out'
    []
  []
  