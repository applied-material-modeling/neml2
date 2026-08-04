[Tensors]
    [C] # MPa
        type = Scalar
        values = '0.0'
        batch_shape = '(1)'
    []
    [g] # unitless
        type = Scalar
        values = '0.0'
        batch_shape = '(1)'
    []
    [k1] # mm^-1
        type = Scalar
        values = '3.0e7'
        batch_shape = '(1)'
    []
    [k2_0] # unitless
        type = Scalar
        values = '6000.0'
        batch_shape = '(1)'
    []
    [Q_d]
        type = Scalar
        values = '0.01'
        batch_shape = '(1)'
    []
    [T_0] # K
        type = Scalar
        values = '3325.5'
        batch_shape = '(1)'
    []
    [Bk] # MPa * s
        type = Scalar
        values = '8.3e-11'
        batch_shape = '(1)'
    []
    [tau_p] # MPa
        type = Scalar
        values = '950.0'
    []
    [H_0] # eV
        type = Scalar
        values = '2.55'
    []
    [alpha]
        type = Scalar
        values = '0.23'
    []
    [p]
        type = Scalar
        values = '0.6'
    []
    [q]
        type = Scalar
        values = '1.4'
    []
    [m]
        type = Scalar
        values = '0.33'
    []
    [a] # mm
        type = Scalar
        values = '3.16e-7'
    []
    [b] # mm
        type = Scalar
        values = '2.73664028e-7'
    []
    [h] # mm
        type = Scalar
        values = '2.5801292e-7'
    []
    [w]
        type = Scalar
        values = '7.9e-6'
    []
    [kB] # eV/K
        type = Scalar
        values = '8.617e-5'
    []
[]

[Models]
    [E]
        type = ScalarQuadraticInterpolation
        a = -2.716e-2
        b = 0.01253e3
        c = 396.507e3
        argument = 'forces/T'
        output = 'E'
    []
    [nu]
        type = ScalarQuadraticInterpolation
        a = 3.157e-9
        b = -8.030e-6
        c = 0.285
        argument = 'forces/T'
        output = 'nu'
    []
    [G_bottom_inner]
        type = ScalarLinearCombination
        from_var = 'nu'
        to_var = 'G_bottom_inner'
        constant_coefficient = 1
    []
    [G_bottom]
        type = ScalarMultiplication
        from_var = 'G_bottom_inner'
        to_var = 'G_bottom'
        coefficient = 2
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
        d_lath = 0.5e-3
        c_block = 0.0
        d_block = 1.0
        c_packet = 0.0
        d_packet = 1.0
        c_PAG = 0.0
        d_PAG = 1.0
        use_L3 = false
        rho_m = 'state/internal/rho_m'
        L = 'state/internal/L'
    []
    [athermal]
        type = AthermalStress
        shear_modulus = 'G'
        alpha = 'alpha'
        b = 'b'
        sigma_ss = 0.0
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
        type = ScalarMultiplication
        from_var = 'state/internal/s'
        to_var = 'state/internal/tau_eff'
        coefficient = 'm'
    []
    [shear_a]
        type = ScalarMultiplication
        from_var = 'state/internal/s_a'
        to_var = 'state/internal/tau_a'
        coefficient = 'm'
    []
    [v_disl]
        type = ThermallyActivatedDislocationMobility
        effective_shear = 'state/internal/tau_eff'
        athermal_shear = 'state/internal/tau_a'
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
    [shear_rate]
        type = OrowanEquation
        dislocation_density = 'state/internal/rho_m'
        v_disl = 'state/internal/v_disl'
        b = 'b'
        plastic_shear_rate = 'state/internal/gamma_rate'
    []
    [flow_rate]
        type = ScalarMultiplication
        from_var = 'state/internal/gamma_rate'
        to_var = 'state/internal/p_rate'
        coefficient = 'm'
    []
    [rho_m_rate]
        type = KocksMeckingDislocationDensity
        plastic_flow_rate = 'state/internal/p_rate'
        k1 = 'k1'
        L = 'state/internal/L'
        k2_0 = 'k2_0'
        Q_d = 'Q_d'
        k_B = 'kB'
        temperature = 'forces/T'
        dislocation_density = 'state/internal/rho_m'
        density_rate = 'state/internal/rho_m_rate'
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
        models = 'G_bottom_inner G_bottom G mandel_stress kinharden overstress vonmises L athermal normality shear_eff shear_a v_disl shear_rate rho_m_rate flow_rate Eprate Erate Eerate elasticity integrate_rho_m integrate_stress integrate_X mixed mixed_old rename'
    []
[]