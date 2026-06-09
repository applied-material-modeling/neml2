[Drivers]
    [unit]
        type = ModelUnitTest
        model = 'model'
        input_Scalar_names = 'state/internal/L'
        input_Scalar_values = '3.16227766e-7'
        output_Scalar_names = 'state/internal/s_a'
        output_Scalar_values = '69.29974077'
        check_AD_parameter_derivatives = true
    []
[]

[Models]
    [model]
        type = AthermalStress
        shear_modulus = 160156.25
        alpha = 0.5
        b = 2.73664028e-10
        L = 'state/internal/L'
        athermal_stress = 'state/internal/s_a'
    []
[]
