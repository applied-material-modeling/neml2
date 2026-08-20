[Drivers]
    [unit]
        type = ModelUnitTest
        model = 'model'
        input_Scalar_names = 'state/internal/gamma_rate state/internal/L forces/T state/internal/rho_m'
        input_Scalar_values = '10 1.0 573.15 1.0e12'
        output_Scalar_names = 'state/internal/rho_m_rate'
        output_Scalar_values = '2.99903464e11'
        derivative_rel_tol = 1e-4
        parameter_derivative_rel_tol = 1e-4
    []
[]

[Models]
    [model]
        type = KocksMeckingDislocationDensity
        plastic_flow_rate = 'state/internal/gamma_rate'
        k1 = 3.0e10
        L = 'state/internal/L'
        k2_0 = 6000
        Q_d = 1
        k_B = 8.617e-5
        temperature = 'forces/T'
        dislocation_density = 'state/internal/rho_m'
        density_rate = 'state/internal/rho_m_rate'
    []
[]