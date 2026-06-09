[Drivers]
    [unit]
        type = ModelUnitTest
        model = 'model'
        input_Scalar_names = 'state/internal/tau_eff state/internal/tau_a forces/T'
        input_Scalar_values = '100.0 44.8497 523.15'
        output_Scalar_names = 'state/internal/v_disl'
        output_Scalar_values = '3.1740e-08'
        check_AD_parameter_derivatives = false
    []
[]

[Models]
    [model]
        type = ThermallyActivatedDislocationMobility
        effective_shear = 'state/internal/tau_eff'
        athermal_shear = 'state/internal/tau_a'
        temperature = 'forces/T'
        h = 2.5801e-10
        w = 7.9e-9
        b = 2.7366e-10
        a = 3.16e-10
        Bk = 4.15e-8
        tau_p = 2.03e3
        T_0 = 2956
        p = 0.86
        q = 1.69
        k_B = 8.617e-5
        H_0 = 1.63
        v_disl = 'state/internal/v_disl'
    []
[]