[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_scalar_names = 'state/delta state/r1 params/rho_rat params/D'
      input_scalar_values = 'delta rr rho_rat D'
      output_scalar_names = 'state/delta_growth'
      output_scalar_values = 'delta_growth'
      check_derivatives = true
      derivative_rel_tol = 1e-4
  []
[]

[Tensors]
    [delta]
        type = Scalar
        values = "0.1 0.5 0.8"
        batch_shape = '(3)'
    []
    [rr]
        type = Scalar
        values = "0.7 0.8 0.2"
        batch_shape = '(3)'
    []
    [rho_rat]
        type = Scalar
        values = "0.1 0.8 0.3"
        batch_shape = '(3)'
    []
    [D]
        type = Scalar
        values = "0.0001 0.235 0.0556"
        batch_shape = '(3)'
    []
    [delta_growth]
        type = Scalar
        values = "0.1240102597 22.47772766 0.1335548307"
        batch_shape = '(3)'
    []
[]

[Models]
    [dense_ratio]
        type = ScalarInputParameter
        from = 'params/rho_rat'
    []
    [diffusion]
        type = ScalarInputParameter
        from = 'params/D'
    []
    [model0]
        type = LiquidProductDiffusion1D
        Liquid_Product_Density_Ratio = 'dense_ratio'
        Initial_Porosity = 0.5
        Product_Thickness_Growth_Ratio = 0.576
        Liquid_Product_Diffusion_Coefficient = 'diffusion'
        Representative_Pores_Size = 0.2

        Inlet_Gap = 'state/r1'
        Product_Thickness = 'state/delta'
        Ideal_Thickness_Growth = 'state/delta_growth'
    []
    [model]
        type = ComposedModel
        models = 'model0'
    []
[]