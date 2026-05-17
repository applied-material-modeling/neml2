[Drivers]
    [unit]
        type = ModelUnitTest
        model = 'model'
        input_Scalar_names = 'state/internal/rho_m'
        input_Scalar_values = 1.0e12
        output_Scalar_names = 'state/internal/L'
        output_Scalar_values = 1.0e-6
    []
[]

[Models]
    [model]
        type = MeanFreePath
        use_L2 = false
        use_L3 = false
        rho_m = 'state/internal/rho_m'
        L = 'state/internal/L'
    []
[]