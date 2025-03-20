nbatch = '(1)'
nstep = 100

# denisty kgm-3
rho_s = 320
rho_b = 1250 # 1.2 and 1.4
rho_g = 1
rho_p = 3210

rho_sm1 = 3.125e-3
rho_bm1 = 8e-4 # 1.2 and 1.4
rho_gm1 = 1
rho_pm1 = 0.00311526479

# heat capacity Jkg-1K-1
cp_s = 1592
cp_b = 1200
cp_g = 1e-4
cp_p = 750

# thermal conductivity W/m-1K-1
k_s = 1.5 
k_b = 0.279
k_g = 1e-4
k_p = 380 #120 and 490

# reaction type
Ea = 177820 # J mol-1
A = 5.24e12 # s-1
R = 8.31446261815324 # JK-1mol-1

Y = 0.7
Ym1 = 1.428571429 # 1/Y
invYm1 = -0.4285714285 # 1-1/Y

order = 1.0
k = 1.0

# initial mass kg
ms0 = 0.5 
mb0 = 10
mp0 = 25
mg0 = 1e-5
alpha0 = 3.333333 # 1/(1-Y)

[Tensors]
    ############### Run condition ############### 
    [endtime]
        type = Scalar
        values = '100'
        batch_shape = '${nbatch}'
    []
    [times]
        type = LinspaceScalar
        start = 0
        end = endtime
        nstep = '${nstep}'
    []
    [endT]
        type = Scalar
        values = '1500'
        batch_shape = '${nbatch}'
    []
    [T]
        type = LinspaceScalar
        start = 0
        end = endT
        nstep = '${nstep}'
    []
    ############### Simulation parameters ############### 
    [Ea]
        type = Scalar
        values = '${Ea}'
        batch_shape = '${nbatch}'
    []
    [A]
        type = Scalar
        values = '${A}'
        batch_shape = '${nbatch}'
    []
    [order]
        type = Scalar
        values = '${order}'
        batch_shape = '${nbatch}'
    []
    [k]
        type = Scalar
        values = '${k}'
        batch_shape = '${nbatch}'
    []
[]

[Drivers]
    [driver]
        type = PyrolysisDriver
        model = 'model'
        prescribed_time = 'times'
        time = 'forces/tt'

        prescribed_temperature = 'T'
        temperature = 'forces/T'
        
        show_input_axis = true
        show_output_axis = true
        show_parameters = true
        
        ic_Scalar_names = 'state/mp state/mb state/mg state/ms state/alpha'
        ic_Scalar_values = '${mp0} ${mb0} ${mg0} ${ms0} ${alpha0}'
        save_as = 'test.pt'

        verbose = true
    []
    [regression]
        type = TransientRegression
        driver = 'driver'
        reference = 'gold/result.pt'
    []
[]

[Solvers]
    [newton]
        #type = NewtonWithLineSearch
        #linesearch_type = STRONG_WOLFE
        #linesearch_cutback = 1.55
        
        type = Newton
        rel_tol = 1e-8
        abs_tol = 1e-10
        max_its = 100
        verbose = true
    []
[]

[Models]
    [amount]
        type = PyrolysisConversionAmount
        initial_mass_solid = '${ms0}'
        initial_mass_binder = '${mb0}'
        reaction_yield = '${Y}'

        mass_solid = 'state/ms'
        reaction_amount = 'state/alpha'
    []
    [reaction]
        type = ChemicalReactionModel
        scaling_constant = 'k'
        reaction_order = 'order'
        reaction_amount = 'state/alpha'
        reaction_out = 'state/f'
    []
    [pyrolysis]
        type = PyrolysisKinetics
        kinetic_constant = 'A'
        activation_energy = 'Ea'
        ideal_gas_constant = '${R}'
        temperature = 'forces/T'
        reaction = 'state/f'
        out = 'state/pyro'
    []
    [amount_rate]
        type = ScalarVariableRate
        variable = 'state/alpha'
        time = 'forces/tt'
        rate = 'state/alpha_dot'
    []
    [residual_ms]
        type = ScalarLinearCombination
        coefficients = "1.0 -1.0"
        from_var = 'state/alpha_dot state/pyro'
        to_var = 'residual/ms'   
    []
    [rms]
        type = ComposedModel
        models = 'amount reaction pyrolysis amount_rate residual_ms'
    []
    [solid_rate]
        type = ScalarVariableRate
        variable = 'state/ms'
        time = 'forces/tt'
        rate = 'state/ms_dot'
    []
    [binder_rate]
        type = ScalarVariableRate
        variable = 'state/mb'
        time = 'forces/tt'
        rate = 'state/mb_dot'
    []
    [residual_binder]
        type = ScalarLinearCombination
        coefficients = "1.0 ${Ym1}"
        from_var = 'state/mb_dot state/ms_dot'
        to_var = 'residual/mb'
    []
    [rmb]
        type = ComposedModel
        models = 'solid_rate binder_rate residual_binder'
    []
    [gas_rate]
        type = ScalarVariableRate
        variable = 'state/mg'
        time = 'forces/tt'
        rate = 'state/mg_dot'
    []
    [residual_gas]
        type = ScalarLinearCombination
        coefficients = "1.0 ${invYm1}"
        from_var = 'state/mg_dot state/ms_dot'
        to_var = 'residual/mg'
    []
    [rmg]
        type = ComposedModel
        models = 'solid_rate gas_rate residual_gas'
    []
    [rmp]
        type = ScalarLinearCombination
        coefficients = "1.0 -1.0"
        from_var = 'state/mp old_state/mp'
        to_var = 'residual/mp'
    []
    [model_residual]
        type = ComposedModel
        models = 'rmp rms rmb rmg'
        automatic_scaling = false
    []
    [model_update]
        type = ImplicitUpdate
        implicit_model = 'model_residual'
        solver = 'newton'
    []
    [amount_new]
        type = PyrolysisConversionAmount
        initial_mass_solid = '${ms0}'
        initial_mass_binder = '${mb0}'
        reaction_yield = '${Y}'

        mass_solid = 'state/ms'
        reaction_amount = 'state/alpha'
    []
    [model_solver]
        type = ComposedModel
        models = 'amount_new model_update'
        additional_outputs = 'state/ms'
    []
################################### POST PROCESS #################################
#########
######### weight fraction
    [M]
        type = ScalarLinearCombination
        coefficients = "1.0 1.0 1.0 1.0"
        from_var = 'state/mp state/mb state/mg state/ms'
        to_var = 'state/M'
    []
    [wb]
        type = Ratio
        numerator = 'state/mb'
        denominator = 'state/M'
        out = 'state/wb'
    []
    [wp]
        type = Ratio
        numerator = 'state/mp'
        denominator = 'state/M'
        out = 'state/wp'
    []
    [ws]
        type = Ratio
        numerator = 'state/ms'
        denominator = 'state/M'
        out = 'state/ws'
    []
    [wg]
        type = Ratio
        numerator = 'state/mg'
        denominator = 'state/M'
        out = 'state/wg'
    []
    [wout]
        type = ComposedModel
        models = 'M wb wp ws wg'
    []
    #########
    ######### volume fraction
    [Vb]
        type = ScalarLinearCombination
        coefficients = '${rho_bm1}'
        from_var = "state/mb"
        to_var = "state/Vb"
    []
    [Vp]
        type = ScalarLinearCombination
        coefficients = '${rho_pm1}'
        from_var = "state/mp"
        to_var = "state/Vp"
    []
    [Vg]
        type = ScalarLinearCombination
        coefficients = '${rho_gm1}'
        from_var = "state/mg"
        to_var = "state/Vg"
    []
    [Vs]
        type = ScalarLinearCombination
        coefficients = '${rho_sm1}'
        from_var = "state/ms"
        to_var = "state/Vs"
    []
    [V]
        type = ScalarLinearCombination
        coefficients = "1.0 1.0 1.0 1.0"
        from_var = 'state/Vp state/Vb state/Vg state/Vs'
        to_var = 'state/V'
    []
    [vb]
        type = Ratio
        numerator = 'state/Vb'
        denominator = 'state/V'
        out = 'state/vb' 
    []
    [vp]
        type = Ratio
        numerator = 'state/Vp'
        denominator = 'state/V'
        out = 'state/vp' 
    []
    [vg]
        type = Ratio
        numerator = 'state/Vg'
        denominator = 'state/V'
        out = 'state/vg' 
    []
    [vs]
        type = Ratio
        numerator = 'state/Vs'
        denominator = 'state/V'
        out = 'state/vs' 
    []
    [vout]
        type = ComposedModel
        models = 'V Vp Vb Vg Vs vb vp vs vg'
    []
    #########
    ######### element properties
    [rho]
        type = ScalarLinearCombination
        coefficients = "${rho_p} ${rho_b} ${rho_g} ${rho_s}"
        from_var = 'state/vp state/vb state/vg state/vs'
        to_var = 'state/rho'
    []
    [cp]
        type = ScalarLinearCombination
        coefficients = "${cp_p} ${cp_b} ${cp_g} ${cp_s}"
        from_var = 'state/wp state/wb state/wg state/ws'
        to_var = 'state/cp'
    []
    [K]
        type = ScalarLinearCombination
        coefficients = "${k_p} ${k_b} ${k_g} ${k_s}"
        from_var = 'state/vp state/vb state/vg state/vs'
        to_var = 'state/K'
    []
    [V_nogas]
        type = ScalarLinearCombination
        coefficients = "1.0 1.0 1.0"
        from_var = 'state/Vp state/Vb state/Vs'
        to_var = 'state/V_nogas'
    []
    [elout]
        type = ComposedModel
        models = 'Vp Vb Vs Vg V vp vb vg vs wout rho cp K V_nogas'
    []
    [model]
        type = ComposedModel
        models = 'model_solver vout wout elout'
        additional_outputs = 'state/ms state/mp state/mg state/mb'
    []
[]