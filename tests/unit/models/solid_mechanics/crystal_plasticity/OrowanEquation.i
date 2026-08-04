[Drivers]
    [unit]
        type = ModelUnitTest
        model = 'model'
        input_Scalar_names = 'state/internal/rho_m state/internal/v_disl'
        input_Scalar_values = '5.0 113254.9002'
        output_Scalar_names = 'state/internal/gamma_rate'
        output_Scalar_values = '1.41568625e-4'
    []
[]

[Models]
    [model]
        type = OrowanEquation
        dislocation_density = 'state/internal/rho_m'
        v_disl = 'state/internal/v_disl'
        b = 2.5e-10
        plastic_shear_rate = 'state/internal/gamma_rate'
    []
[]

