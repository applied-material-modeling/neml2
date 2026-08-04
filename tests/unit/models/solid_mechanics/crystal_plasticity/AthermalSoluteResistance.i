[Drivers]
    [unit]
        type = ModelUnitTest
        model = 'model'
        input_Scalar_names = 'state/internal/L'
        input_Scalar_values = '1.0'
        output_Scalar_names = 'state/internal/s_0'
        output_Scalar_values = '2.19145022e-5'
        check_AD_parameter_derivatives = true
    []
[]

[Models]
    [model]
        type = AthermalSoluteResistance
        shear_modulus = 160156.25
        alpha = 0.5
        b = 2.73664028e-10
        include_solid_solution = false
        L = 'state/internal/L'
        athermal_solute_resistance = 'state/internal/s_0'
    []
[]
