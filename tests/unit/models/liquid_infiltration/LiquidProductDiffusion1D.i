[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/delta state/r1 params/rho_rat params/D'
      input_Scalar_values = 'delta rr rho_rat D'
      output_Scalar_names = 'state/delta_growth'
      output_Scalar_values = 'delta_growth'
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
    [M]
        type = Scalar
        values = "0.576"
        batch_shape = '(3)'
    []
    [phi0]
        type = Scalar
        values = "0.5"
        batch_shape = '(3)'
    []
    [lc]
        type = Scalar
        values = "0.2"
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
        liquid_product_density_ratio = 'dense_ratio'
        initial_porosity = 'phi0'
        product_thickness_growth_ratio = 'M'
        liquid_product_diffusion_coefficient = 'diffusion'
        representative_pores_size = 'lc'

        inlet_gap = 'state/r1'
        product_thickness = 'state/delta'
        ideal_thickness_growth = 'state/delta_growth'
    []
    [model]
        type = ComposedModel
        models = 'model0'
    []
[]