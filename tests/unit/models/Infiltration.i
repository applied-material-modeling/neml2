[Drivers]
    [unit]
        type = ModelUnitTest
        model = 'model'
        batch_shape = '(6)'
        input_scalar_names = 'state/delta state/h state/alpha old_state/delta old_state/h old_state/alpha forces/alpha_in forces/time old_forces/time'
        input_scalar_values = 'delta h alpha deltan hn alphan aI t tn'
        output_scalar_names = 'residual/delta residual/h residual/alpha'
        output_scalar_values = 'rdelta_expected rh_expected rstate_expected'
        check_second_derivatives = false
        check_AD_first_derivatives = true
        check_AD_second_derivatives = false
        check_AD_derivatives = false
    []
[]

[Tensors]
    [delta]
        type = Scalar
        values = "0.001 0.001 0.45 0.65 0.99 0.99"
        batch_shape = '(6)'
    []
    [deltan]
        type = Scalar
        values = "0.0001 0.0001 0.15 0.15 0.89 0.89"
        batch_shape = '(6)'
    []
    [h]
        type = Scalar
        values = "0.001 0.99 0.65 0.45 0.001 0.99"
        batch_shape = '(6)'
    []
    [hn]
        type = Scalar
        values = "0.0001 0.89 0.15 0.15 0.0001 0.89"
        batch_shape = '(6)'
    []
    [alpha]
        type = Scalar
        values = "10000.0 10000.0 15000.0 5000.0 2000.0 1.0"
        batch_shape = '(6)'
    []
    [alphan]
        type = Scalar
        values = "10000.0 10000.0 15000.0 5000.0 2000.0 1.0"
        batch_shape = '(6)'
    []
    [aI]
        type = Scalar
        values = "0.3 0.2 0.2 0.2 0.0 0.0"
        batch_shape = '(6)'
    []
    [t]
        type = Scalar
        values = "108.0 108.0 108.0 108. 108.0 108.08"
        batch_shape = '(6)'
    []
    [tn]
        type = Scalar
        values = "8.0 8.0 8.0 8.0 8.0 8.0"
        batch_shape = '(6)'
    []
    ## expected residual value
    [rdelta_expected]
        type = Scalar
        values = "1.0 1.0 1.0 1.0 1.0 1.0"
        batch_shape = '(6)'
    []
    [rh_expected]
        type = Scalar
        values = "1.0 1.0 1.0 1.0 1.0 1.0"
        batch_shape = '(6)'
    []
    [rstate_expected]
        type = Scalar
        values = "1.0 1.0 1.0 1.0 1.0 1.0"
        batch_shape = '(6)'
    []
[]

[Models]
    [model]
        type = LiquidInfiltration

        ### field variables
        alpha_liquid_inlet_rate = 'forces/alpha_in'
        time = 'forces/time'
        product_thickness = 'state/delta'
        product_height = 'state/h'
        liquid_concentration = 'state/alpha'
        residual_delta = 'residual/delta'
        residual_h = 'residual/h'
        residual_alpha = 'residual/alpha'

        ### RVE properties
        Product_Thickness_Growth_Ratio = 1.0
        RVE_characteristic_length = 1.0
        Product_Deficient_Height_Growth = 1.0
        Initial_Porosity = 0.5
        Liquid_Molar_Volume = 0.000012
        Product_Molar_Volume = 0.0000125
        Liquid_Product_Diffusion_Coefficients = 0.0000005
        Liquid_Solid_Density_Ratio = 0.8
        Sigmoid_smooth_degree = 100.0
    []
[]