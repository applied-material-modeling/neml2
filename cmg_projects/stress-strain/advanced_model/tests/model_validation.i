[Tensors]
    [C] # MPa
        type = Scalar
        values = 0
        batch_shape = '(1)'
    []
    [g] # unitless
        type = Scalar
        values = 0
        batch_shape = '(1)'
    []
    [k1] # m^-1
        type = Scalar
        values = 0
        batch_shape = '(1)'
    []
    [k2] # unitless
        type = Scalar
        values = 0
        batch_shape = '(1)'
    []
    [S]
        type = Scalar
        values = 0
        batch_shape = '(1)'
    []
    [tau_p] # MPa
        type = Scalar
        values = 2030
    []
    [H_0] #eV
        type = Scalar
        values = 1.63
    []
    [alpha]
        type = Scalar
        values = 0
    []
    [p]
        type = Scalar
        values = 0.86
    []
    [q]
        type = Scalar
        values = 1.69
    []
    [Bk] # MPa*s
        type = Scalar
        values = 8.3e-11
    []
    [T_0] #K
        type = Scalar
        values = 2956
    []
    [a] #m
        type = Scalar
        values = 3.16e-10
    []
    [b] #m
        type = Scalar
        values = 2.737e-10
    []
    [h] #m
        type = Scalar
        values = 2.581e-10
    []
    [k_B] # eV/K
        type = Scalar
        values = 8.617e-5
    []
    [m]
        type = Scalar
        values = 0.333
    []
    [E] #MPa
        type = Scalar
        values = 410000
    []
    [nu]
        type = Scalar
        values = 0.28
    []
    [G]
        type = Scalar
        values = 160156.25
    []
[]

[Models]
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
        k_B = 'k_B'
        s = 'S'
        H_0 = 'H_0'
        v_disl = 'state/internal/v_disl'
    []
    [v_disl_diag]
        type = ThermallyActivatedDislocationMobility_diag
        effective_shear      = 'state/internal/tau_eff'
        athermal_shear       = 'state/internal/s_a'
        dislocation_density  = 'state/internal/rho_m'
        temperature          = 'forces/T'
        h    = 'h'
        b    = 'b'
        a    = 'a'
        Bk   = 'Bk'
        tau_p = 'tau_p'
        T_0  = 'T_0'
        p    = 'p'
        q    = 'q'
        k_B  = 'k_B'
        s    = 'S'
        H_0  = 'H_0'
        v_disl    = 'state/internal/v_disl'
        tau_ratio = 'state/internal/tau_ratio'
        D_G       = 'state/internal/D_G'
        mclD_G    = 'state/internal/mclD_G'
        exp_arg   = 'state/internal/exp_arg'
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
        models = 'mandel_stress kinharden overstress vonmises athermal normality shear_eff v_disl rho_m_rate flow_rate Eprate Erate Eerate elasticity integrate_rho_m integrate_stress integrate_X mixed mixed_old rename'
    []
[]
