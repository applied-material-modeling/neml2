[Tensors]
    [T_train]
        type = Scalar
        values = '522.15 573.15 724.15'
        batch_shape = '(3)'
        intermediate_dimension = 1
    []
    [T_values]
        type = Scalar
        values = '523.15 573.15 723.15'
        batch_shape = '(3)'
        intermediate_dimension = 1
    []
    [C_values] # MPa
        type = Scalar
        values = '5.0e3 5.0e3 5.0e3'
        batch_shape = '(3)'
        intermediate_dimension = 1
    []
    [g_values] # unitless
        type = Scalar
        values = '30.0 30.0 30.0'
        batch_shape = '(3)'
        intermediate_dimension = 1
    []
    [k1_values] # microns^-1
        type = Scalar
        values = '5.0 5.0 5.0'
        batch_shape = '(3)'
        intermediate_dimension = 1
    []
    [k2_values] # unitless
        type = Scalar
        values = '3.5 3.5 3.5'
        batch_shape = '(3)'
        intermediate_dimension = 1
    []
    [T_0_values] # K
        type = Scalar
        values = '249.9387 249.9387 249.9387'
        batch_shape = '(3)'
        intermediate_dimension = 1
    []
    [Bk_values] # MPa * s
        type = Scalar
        values = '6.6e-6 6.6e-6 6.6e-6'
        batch_shape = '(3)'
        intermediate_dimension = 1
    []
    [S_values]
        type = Scalar
        values = '30.0 30.0 30.0'
        batch_shape = '(3)'
        intermediate_dimension = 1
    []
    [tau_p] # MPa
        type = Scalar
        values = '360'
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
    [a] # microns
        type = Scalar
        values = '2.868e-4'
        batch_shape = '(1)'
    []
    [b] # microns
        type = Scalar
        values = '2.483760858e-4'
        batch_shape = '(1)'
    []
    [h] # microns
        type = Scalar
        values = '2.703976331e-4'
        batch_shape = '(1)'
    []
    [kB] # eV/K
        type = Scalar
        values = '8.617e-5'
        batch_shape = '(1)'
    []
[]

[Models]
    [C]
        type = ScalarLinearInterpolation
        argument = 'forces/T'
        abscissa = 'T_train'
        ordinate = 'C_values'
    []
    [g]
        type = ScalarLinearInterpolation
        argument = 'forces/T'
        abscissa = 'T_train'
        ordinate = 'g_values'
    []
    [k1]
        type = ScalarLinearInterpolation
        argument = 'forces/T'
        abscissa = 'T_train'
        ordinate = 'k1_values'
    []
    [k2]
        type = ScalarLinearInterpolation
        argument = 'forces/T'
        abscissa = 'T_train'
        ordinate = 'k2_values'
    []
    [T_0]
        type = ScalarLinearInterpolation
        argument = 'forces/T'
        abscissa = 'T_train'
        ordinate = 'T_0_values'
    []
    [Bk]
        type = ScalarLinearInterpolation
        argument = 'forces/T'
        abscissa = 'T_train'
        ordinate = 'Bk_values'
    []
    [S]
        type = ScalarLinearInterpolation
        argument = 'forces/T'
        abscissa = 'T_train'
        ordinate = 'S_values'
    []
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
    [athermal]
        type = AthermalStress
        shear_modulus = 'G'
        alpha = 'alpha'
        b = 'b'
        dislocation_density = 'state/internal/rho_m'
        athermal_stress = 'state/internal/s_a'
    []
    [yield]
        type = YieldFunction
        yield_stress = '0.0'
    []
    [full_yield]
        type = ScalarLinearCombination
        from_var = 'state/internal/fp state/internal/s_a'
        to_var = 'state/internal/fp_n'
        coefficients = '1 -1'
    []
    [flow]
        type = ComposedModel
        models = 'overstress vonmises yield full_yield'
    []
    [normality]
        type = Normality
        model = 'flow'
        function = 'state/internal/fp_n'
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
        dislocation_density = 'state/internal/rho_m'
        temperature = 'forces/T'
        h = 'h'
        b = 'b'
        a = 'a'
        Bk = 'Bk'
        tau_p = 'tau_p'
        T_0 = 'T_0'
        p = 'p'
        q = 'q'
        k_B = 'kB'
        s = 'S'
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
        models = 'G_bottom G_bottom_inner mandel_stress kinharden overstress vonmises athermal normality shear_eff v_disl rho_m_rate flow_rate Eprate Erate Eerate elasticity integrate_rho_m integrate_stress integrate_X mixed mixed_old rename'
    []
[]