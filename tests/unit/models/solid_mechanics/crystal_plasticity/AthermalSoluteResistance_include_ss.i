[Drivers]
    [unit]
        type = ModelUnitTest
        model = 'model'
        input_Scalar_names = 'state/internal/L forces/T state/internal/v_disl'
        input_Scalar_values = '1.0 573.15 5.53064e-7'
        output_Scalar_names = 'state/internal/s_0'
        output_Scalar_values = '2.51396471e7'
        check_AD_parameter_derivatives = true
    []
[]

[Models]
    [model]
        type = AthermalSoluteResistance
        shear_modulus = 160156.25
        alpha = 0.5
        b = 2.73664028e-10
        L = 'state/internal/L'
        include_solid_solution = true
        t_a0 = 1e-9
        Q_a = 2.5635e-19
        k_B = 1.380649e-23
        temperature = 'forces/T'
        m = 0.33
        v_disl = 'state/internal/v_disl'
        tau_s0 = 130e6
        p_ss = 0.37
        athermal_solute_resistance = 'state/internal/s_0'
    []
[]
