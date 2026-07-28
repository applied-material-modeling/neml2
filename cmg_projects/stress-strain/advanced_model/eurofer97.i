[Tensors]
    [C_values] # MPa
        type = Scalar
        values = '19.0e3'
        batch_shape = '(1)'
    []
    [g_values] # unitless
        type = Scalar
        values = '125.0'
        batch_shape = '(1)'
    []
    [k1_values] # mm^-1
        type = Scalar
        values = '7.8e7'
        batch_shape = '(1)'
    []
    [k2_values] # unitless
        type = Scalar
        values = '3.5'
        batch_shape = '(1)'
    []
    [T_0_values] # K
        type = Scalar
        values = '249.9387'
        batch_shape = '(1)'
    []
    [Bk_values] # MPa * s
        type = Scalar
        values = '6.6e-11'
        batch_shape = '(1)'
    []
    [tau_p] # MPa
        type = Scalar
        values = '380'
        batch_shape = '(1)'
    []
    [H_0] # eV
        type = Scalar
        values = '0.65'
        batch_shape = '(1)'
    []
    [alpha]
        type = Scalar
        values = '0.5'
        batch_shape = '(1)'
    []
    [p]
        type = Scalar
        values = '0.5'
        batch_shape = '(1)'
    []
    [q]
        type = Scalar
        values = '1.25'
        batch_shape = '(1)'
    []
    [m]
        type = Scalar
        values = '0.333'
        batch_shape = '(1)'
    []
    [a] # mm
        type = Scalar
        values = '2.868e-7'
        batch_shape = '(1)'
    []
    [b] # mm
        type = Scalar
        values = '2.483760858e-7'
        batch_shape = '(1)'
    []
    [h] # mm
        type = Scalar
        values = '2.703976331e-7'
        batch_shape = '(1)'
    []
    [w] # mm
        type = Scalar
        values = '7.17e-6'
    []
    [kB] # eV/K
        type = Scalar
        values = '8.617e-5'
        batch_shape = '(1)'
    []
[]

[Models]
    [E]
        type = ScalarQuadraticInterpolation
        a = '-7.626e-2'
        b = '0.01879e3'
        c = '207968'
        argument = 'forces/T'
        output = 'E'
    []
    [nu]
        type = ScalarQuadraticInterpolation
        a = '1.609e-9'
        b = '-4.449e-5'
        c = '0.302'
        argument = 'forces/T'
        output = 'nu'
    []
    [G_bottom_inner]
        type = ScalarLinearCombination
        from_var = 'nu'
        to_var = 'G_bottom_inner'
        constant_coefficient = '1'
    []
    [G_bottom]
        type = ScalarMultiplication
        from_var = 'G_bottom_inner'
        to_var = 'G_bottom'
        coefficient = '2'
    []
    [G]
        type = ScalarMultiplication
        from_var = 'E G_bottom'
        to_var = 'G'
        reciprocal = 'false true'
    []
    [mandel_stress]
        type = IsotropicMandelStress
    []
    [kinharden]
        type = FredrickArmstrongPlasticHardening
        C = 'C'
        g = 'g'
    []
    [overstress]
        type = SR2LinearCombination
        from_var = 'state/internal/M state/internal/X'
        to_var = 'state/internal/O'
        coefficients = '1 -1'
    []
    [vonmises]
        type = SR2Invariant
        invariant_type = 'VONMISES'
        tensor = 'state/internal/O'
        invariant = 'state/internal/s'
    []
    [L]
        type = MeanFreePath
        use_L2 = true
        c_lath = 1.0
        d_lath = 0.0005
        c_block = 0.3
        d_block = 0.0031
        c_packet = 0.0
        d_packet = 0.01
        c_PAG = 0.1
        d_PAG = 0.0070
        use_L3 = true
        c_MX = 0.3
        d_MX = 0.01
        c_M23C6 = 0.2
        d_M23C6 = 0.01
        rho_m = 'state/internal/rho_m'
        L = 'state/internal/L'
    []
    [athermal]
        type = AthermalStress
        shear_modulus = 'G'
        alpha = 'alpha'
        b = 'b'
        L = 'state/internal/L'
        athermal_stress = 'state/internal/s_a'
    []
    [yield]
        type = YieldFunction
        yield_stress = '0.0'
        isotropic_hardening = 'state/internal/s_a'
    []
    [flow]
        type = ComposedModel
        models = 'overstress vonmises yield'
    []
    [normality]
        type = Normality
        model = 'flow'
        function = 'state/internal/fp'
        from = 'state/internal/M'
        to = 'state/internal/NM'
    []
    [shear_eff]
        type = NormalToShearStress
        normal_stress = 'state/internal/s'
        shear_stress = 'state/internal/tau_eff'
        schmid_factor = 'm'
    []
    [v_disl]
        type = ThermallyActivatedDislocationMobility
        effective_shear = 'state/internal/tau_eff'
        athermal_shear = 'state/internal/s_a'
        temperature = 'forces/T'
        h = 'h'
        w = 'w'
        b = 'b'
        a = 'a'
        Bk = 'Bk'
        tau_p = 'tau_p'
        T_0 = 'T_0'
        p = 'p'
        q = 'q'
        k_B = 'kB'
        H_0 = 'H_0'
        v_disl = 'state/internal/v_disl'
    []
    [rho_m_rate]
        type = KocksMeckingDislocationDensity
        plastic_flow_rate = 'state/internal/gamma_rate'
        k1 = 'k1'
        k2 = 'k2'
        dislocation_density = 'state/internal/rho_m'
        density_rate = 'state/internal/rho_m_rate'
    []
    [flow_rate]
        type = OrowanEquation
        dislocation_density = 'state/internal/rho_m'
        v_disl = 'state/internal/v_disl'
        b = 'b'
        plastic_flow_rate = 'state/internal/gamma_rate'
    []
    [Eprate]
        type = AssociativePlasticFlow
    []
    [Erate]
        type = SR2VariableRate
        variable = 'forces/E'
        rate = 'forces/E_rate'
    []
    [Eerate]
        type = SR2LinearCombination
        from_var = 'forces/E_rate state/internal/Ep_rate'
        to_var = 'state/internal/Ee_rate'
        coefficients = '1 -1'
    []
    [elasticity]
        type = LinearIsotropicElasticity
        coefficients = 'E nu'
        coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
        rate_form = true
    []
    [integrate_rho_m]
        type = ScalarBackwardEulerTimeIntegration
        variable = 'state/internal/rho_m'
    []
    [integrate_stress]
        type = SR2BackwardEulerTimeIntegration
        variable = 'state/S'
    []
    [integrate_X]
        type = SR2BackwardEulerTimeIntegration
        variable = 'state/internal/X'
    []
    [mixed]
        type = MixedControlSetup
        above_variable = 'state/S'
        below_variable = 'forces/E'
    []
    [mixed_old]
        type = MixedControlSetup
        control = 'old_forces/control'
        mixed_state = 'old_state/mixed_state'
        fixed_values = 'old_forces/fixed_values'
        above_variable = 'old_state/S'
        below_variable = 'old_forces/E'
    []
    [rename]
        type = CopySR2
        from = 'residual/S'
        to = 'residual/mixed_state'
    []
    [implicit_rate]
        type = ComposedModel
        models = 'G_bottom G_bottom_inner mandel_stress kinharden overstress vonmises L athermal normality shear_eff v_disl rho_m_rate flow_rate Eprate Erate Eerate elasticity integrate_rho_m integrate_stress integrate_X mixed mixed_old rename'
    []
[]