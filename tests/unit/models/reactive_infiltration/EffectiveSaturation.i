[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/phi state/phimax'
      input_Scalar_values = '0.053456 0.374'
      output_Scalar_names = 'state/out'
      output_Scalar_values = '0.047700534759'
      check_second_derivatives = true
    []
  []
  
  [Models]
    [model]
      type = EffectiveSaturation
      residual_volume_fraction = 0.1
      flow_fraction = 'state/phi'
      max_fraction = 'state/phimax'
      effective_saturation = 'state/out'
    []
  []
  