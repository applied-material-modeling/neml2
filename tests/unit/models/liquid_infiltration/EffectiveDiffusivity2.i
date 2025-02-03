[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/switch state/scale'
      input_Scalar_values = 'switch scale'
      output_Scalar_names = 'state/Deff'
      output_Scalar_values = 'deff'
      check_derivatives = true
      check_AD_parameter_derivatives = false
  []
[]

[Tensors]
    [scale]
        type = Scalar
        values = "25 12 3"
        batch_shape = '(3)'
    []
    [switch]
        type = Scalar
        values = "0.5 0.1 0.9"
        batch_shape = '(3)'
    []
    [deff]
        type = Scalar
        values = "673.8 135.72 1211.88"
        batch_shape = '(3)'
    []
[]

[Models]
    [model0]
        type = EffectiveDiffusivity
        macroscopic_diffusion_coefficient = 1.2
        coefficient_liquid_limit = 1122.0
        scaling_liquid_limit = 'state/scale'
        const_liquid_limit = true
        switching_liquid_limit = 'state/switch'
        effective_diffusion_coefficient = 'state/Deff'
    []
    [model]
        type = ComposedModel
        models = model0
    []
[]