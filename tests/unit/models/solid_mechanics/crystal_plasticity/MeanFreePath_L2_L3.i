[Drivers]
    [unit]
        type = ModelUnitTest
        model = 'model'
        input_Scalar_names = 'state/internal/rho_m'
        input_Scalar_values = 1.0e12
        output_Scalar_names = 'state/internal/L'
        output_Scalar_values = 1.40780568e-8
        check_AD_parameter_derivatives = false
    []
[]

[Models]
    [model]
        type = MeanFreePath
        use_L2 = true
        c_lath = 1.0
        d_lath = 0.5e-6
        c_block = 1.0
        d_block = 3.0e-6
        c_packet = 0.0
        d_packet = 1.0
        c_PAG = 1.0
        d_PAG = 7e-6
        use_L3 = true
        c_MX = 1.0
        d_MX = 18.6e-9
        c_M23C6 = 1.0
        d_M23C6 = 67.6e-9
        rho_m = 'state/internal/rho_m'
        L = 'state/internal/L'
    []
[]