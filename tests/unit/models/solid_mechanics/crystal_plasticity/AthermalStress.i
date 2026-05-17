[Drivers]
    [unit]
        type = ModelUnitTest
        model = 'model'
        input_Scalar_names = 'state/internal/L'
        input_Scalar_values = '1.0e-6'
        output_Scalar_names = 'state/internal/s_a'
        output_Scalar_values = '4.96'
        check_AD_parameter_derivatives = false
    []
[]

[Models]
    [model]
        type = AthermalStress
        shear_modulus = '40000.0'
        alpha = 0.5
        b = 2.48e-10
        L = 'state/internal/L'
        athermal_stress = 'state/internal/s_a'
    []
[]
